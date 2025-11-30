import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

def conv3x3(in_channels, out_channels, stride = 1, padding = 1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, 
                  out_channels=out_channels,
                  kernel_size=3,
                  stride = stride,
                  padding=padding),
        nn.BatchNorm2d(out_channels))
    

def conv1x1(in_channels, out_channels, stride = 1, padding = 1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,
                  out_channels=out_channels,
                  kernel_size=1,
                  stride=stride,
                  padding=padding),
        nn.BatchNorm2d(out_channels)
    )
class Residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, num, stride = 1):
        super().__init__()
        self.conv1 = conv3x3(in_channels=in_channels, out_channels=out_channels,stride=2)
        
        self.relu = nn.ReLU()
        self.conv2 = conv1x1(in_channels=out_channels, out_channels=out_channels//2,padding=0)
        
        self.conv3 = conv3x3(in_channels=out_channels//2, out_channels=out_channels)
        self.num = num

    def forward(self, x):
        out = self.conv1(x)
        
        out = self.relu(out)

        identity = out

        for i in range(self.num):
            out = self.conv2(out)
            out = self.relu(out)
            out = self.conv3(out)
        out += identity
        out = self.relu(out)
        return out

class Darknet53(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = conv3x3(in_channels=3, out_channels=32)
        
        self.relu = nn.ReLU()

        self.layer1 = self._make_layer([32, 64], 1)
        self.layer2 = self._make_layer([64,128], 2)
        self.layer3 = self._make_layer([128,256], 8)
        self.layer4 = self._make_layer([256,512], 8)
        self.layer5 = self._make_layer([512, 1024], 4)

        self.out_filters = (256, 512, 1024)

    def _make_layer(self, channels, nums_blocks):
        layers = []
        layers.append(Residual_block(in_channels=channels[0],
                                     out_channels=channels[1],
                                     num=nums_blocks))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        out3 = x

        x = self.layer4(x)
        out4 = x

        x = self.layer5(x)
        out5 = x

        return out3, out4, out5

class YOLOLayer(nn.Module):
    def __init__(self, anchors, num_classes):
        super().__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.num_anchors = len(self.anchors)
    
    def forward(self, x):
        # Output x shape:[batch, num_anchors*(5+num_classes), grid, grid ]
        batch_size = x.size(0)
        grid_size = x.size(2)

        # 重塑输出维度
        x = x.view(batch_size, self.num_anchors, 5+self.num_classes, grid_size, grid_size)
        x = x.permute(0, 1, 3, 4, 2) # [batch, num_anchors, grid_size, grid_size, 5+num_classes]

        # 解构输出
        # 最后一个维度 5 + num_classes: x, y, w, h, c(置信度), num_classes
        box_centers = torch.sigmoid(x[..., :2]) # x,y 中心点坐标
        box_scales = torch.exp(x[..., 2:4]) # 宽高缩放
        confidence = torch.sigmoid(x[..., 4:5]) # 置信度
        class_probs = torch.sigmoid(x[..., 5:]) # 类别概率

        return torch.cat([box_centers, box_scales, confidence, class_probs], dim=-1)


class YoloV3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes
        anchors = [
            [(116, 90), (156, 198), (373, 326)],
            [(30,61), (62,45), (59,119)],
            [(10,13), (16,30), (33,23)]
        ]
        self.num_anchors = len(anchors)

        # 主干网络
        self.backbone = Darknet53()
        self.small_filters, self.medium_filters, self.large_filters = self.backbone.out_filters 
        self._build_detection_head(filter_nums1=1024, filter_nums2=512, filter_nums3=256)

        self.yolo_layers = nn.ModuleList()
        for anchor in anchors:
            self.yolo_layers.append(YOLOLayer(anchors, self.num_anchors))
    
    def _build_detection_head(self, filter_nums1, filter_nums2, filter_nums3):
        """构建检测头网络"""

        # 检测头1
        self.head1_conv1 = conv1x1(in_channels=self.large_filters, out_channels=filter_nums1//2, padding=0)
        self.relu = nn.ReLU()
        self.head1_conv2 = conv3x3(in_channels=filter_nums1//2, out_channels=filter_nums1)
        self.head1_conv3 = conv1x1(in_channels=filter_nums1, out_channels=filter_nums1//2, padding=0)
        self.head1_conv4 = conv3x3(in_channels=filter_nums1//2, out_channels=filter_nums1)
        self.head1_conv5 = conv1x1(in_channels=filter_nums1, out_channels=filter_nums1//2, padding=0)

        self.head1_conv6 = conv3x3(in_channels=filter_nums1//2, out_channels=filter_nums1)
        self.head1_out = conv1x1(in_channels=filter_nums1,out_channels=self.num_anchors*(self.num_classes+5), padding=0)

        # 上采样
        self.upsample1 = nn.Sequential(
            conv1x1(in_channels=filter_nums1//2, out_channels=256, padding=0),
            nn.Upsample(scale_factor=2.0, mode='nearest')
        )

        # 检测头2

        self.head2_conv1 = conv1x1(in_channels=(self.medium_filters+filter_nums2//2), out_channels=filter_nums2//2, padding=0)
        self.head2_conv2 = conv3x3(in_channels=filter_nums2//2, out_channels=filter_nums2 )
        self.head2_conv3 = conv1x1(in_channels=filter_nums2, out_channels=filter_nums2//2, padding=0)
        self.head2_conv4 = conv3x3(in_channels=filter_nums2//2, out_channels=filter_nums2)
        self.head2_conv5 = conv1x1(in_channels=filter_nums2, out_channels= filter_nums2//2, padding=0)

        self.head2_conv6 = conv3x3(in_channels=filter_nums2//2, out_channels=filter_nums2)
        self.head2_out = conv1x1(in_channels=filter_nums2, out_channels=self.num_anchors*(self.num_classes+5), padding=0)

        # 上采样

        self.upsample2 = nn.Sequential(
            conv1x1(in_channels=filter_nums2//2, out_channels= 128, padding=0),
            nn.Upsample(scale_factor=2.0, mode='nearest')
        )

        # 检测头3

        self.head3_conv1 = conv1x1(in_channels=(self.small_filters + filter_nums3//2), out_channels=filter_nums3//2, padding=0)
        self.head3_conv2 = conv3x3(in_channels=filter_nums3//2, out_channels=filter_nums3)
        self.head3_conv3 = conv1x1(in_channels=filter_nums3, out_channels=filter_nums3//2, padding=0)
        self.head3_conv4 = conv3x3(in_channels=filter_nums3//2, out_channels=filter_nums3)
        self.head3_conv5 = conv1x1(in_channels=filter_nums3, out_channels=filter_nums3//2, padding=0)

        self.head3_conv6 = conv3x3(in_channels=filter_nums3//2, out_channels=filter_nums3)
        self.head3_out = conv1x1(in_channels=filter_nums3, out_channels=self.num_anchors*(self.num_classes + 5), padding=0)

    def forward(self,x):

        # 主干网络特征提取
        small_feat, medium_feat, large_feat = self.backbone(x)

        # 检测头1
        x = self.head1_conv1(large_feat)
        x = self.relu(x)
        x = self.head1_conv2(x)
        x = self.relu(x)
        x = self.head1_conv3(x)
        x = self.relu(x)
        x = self.head1_conv4(x)
        x = self.relu(x)
        x = self.head1_conv5(x)
        route1 = self.relu(x) # 用于上采样

        x = self.head1_conv6(route1)
        x = self.relu(x)
        out1 = self.head1_out(x)

        # 上采样
        x = self.upsample1(route1)
        x = torch.cat([x, medium_feat],1)

        # 检测头2

        x = self.head2_conv1(x)
        x = self.relu(x)
        x = self.head2_conv2(x)
        x = self.relu(x)
        x = self.head2_conv3(x)
        x = self.relu(x)
        x = self.head2_conv4(x)
        x = self.relu(x)
        x = self.head2_conv5(x)
        route2 = self.relu(x)  #  用于上采样

        x = self.head2_conv6(route2)
        x = self.relu(x)
        out2 = self.head2_out(x)

        #上采样
        x = self.upsample2(route2)
        x = torch.cat([x, small_feat], 1)

        # 检测头3

        x = self.head3_conv1(x)
        x = self.relu(x)
        x = self.head3_conv2(x)
        x = self.relu(x)
        x = self.head3_conv3(x)
        x = self.relu(x)
        x = self.head3_conv4(x)
        x = self.relu(x)
        x = self.head3_conv5(x)
        x = self.relu(x)
        x = self.head3_conv6(x)
        x = self.relu(x)
        out3 = self.head3_out(x)

        return out1, out2, out3
        

if __name__ == "__main__":
    input_tensor = torch.randn(1,3,416,416)
    # output = conv3x3(in_channels=32, out_channels=64, stride=2)(input_tensor)
    # x = output
    # output = conv1x1(in_channels=64, out_channels=32)(output)
    # print(output.shape)
    # model = Residual_block(in_channels=128, out_channels=256, num=8)
    # # output = model(input_tensor)
    # # print(output.shape)
    # model = Darknet53()
    # output = model(input_tensor)
    # print(output[0].shape, output[1].shape, output[2].shape)
    # summary(model,(3, 416, 416))

    model = YoloV3(num_anchors=3, num_classes=20)
    out1, out2, out3 = model(input_tensor)
    print(out1.shape, out2.shape, out3.shape)
