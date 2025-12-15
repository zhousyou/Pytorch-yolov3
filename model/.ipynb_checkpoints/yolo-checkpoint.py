import torch
import torch.nn as nn
import torch.nn.functional as F 
import sys
from model.my_darknet import Darknet53, ConvBlock
# import mydarknet.Darknet53 as Darknet53

class UpsampleBlock(nn.Module):
    """
        构建上采样模块
        输入参数：上采样放大的倍数，默认为2
        输出参数：通过上采样之后的特征层
    """
    def __init__(self, scale_factor = 2):
        super().__init__()
        self.scale_factor = 2
    
    def forward(self, x):
        return F.interpolate(input= x,
                             scale_factor=self.scale_factor,
                             mode='nearest')
    

class YoloV3(nn.Module):
    """
        构建完整的yolov3网络模型
    """
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        # 骨干网络 
        # 输出一个列表，包含三个特征层[13,13,1024],[26,26,512],[52,52,256]
        self.darknet = Darknet53()

        # 三个尺度的检测头输出通道数
        # 每个锚点预测(5+num_classes)的值
        # 5代表：4个边界框的坐标 + 1个置信度
        self.num_anchors_per_scale = 3
        self.output_channels = self.num_anchors_per_scale * (5 + self.num_classes)

        # 第一层检测头 13x13
        self.detect_head1 = nn.Sequential(
            ConvBlock(in_channels=1024, out_channels=512, kernel_size=1),
            ConvBlock(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            ConvBlock(in_channels=1024, out_channels=512, kernel_size=1),
            ConvBlock(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            ConvBlock(in_channels=1024, out_channels=512, kernel_size=1)
        )

        self.out_head1 = nn.Sequential(
            ConvBlock(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            # 输出： [batch, 75, 13, 13]
            nn.Conv2d(in_channels=1024, out_channels=self.output_channels, kernel_size=1)
        )
        
        # 连接到第二层的检测头的路径
        self.route1 = ConvBlock(in_channels=512, out_channels=256, kernel_size=1)
        self.upsample1 = UpsampleBlock(scale_factor=2)


        # 第二层检测头 26x26
        self.detect_head2 = nn.Sequential(
            ConvBlock(in_channels=768, out_channels=256, kernel_size=1),
            ConvBlock(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            ConvBlock(in_channels=512, out_channels=256, kernel_size=1),
            ConvBlock(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            ConvBlock(in_channels=512, out_channels=256, kernel_size=1)
        )

        self.out_head2 = nn.Sequential(
            ConvBlock(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            # 输出：[batch, 75, 26, 26]
            nn.Conv2d(in_channels=512, out_channels=self.output_channels, kernel_size=1)
        )

        # 连接到第三层检测头的路径
        self.route2 = ConvBlock(in_channels=256, out_channels=128, kernel_size=1)
        self.upsample2 = UpsampleBlock(scale_factor=2)

        # 第三层检测头 52x52
        self.detect_head3 = nn.Sequential(
            ConvBlock(in_channels=384, out_channels=128, kernel_size=1),
            ConvBlock(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            ConvBlock(in_channels=256, out_channels=128, kernel_size=1),
            ConvBlock(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            ConvBlock(in_channels=256, out_channels=128, kernel_size=1)
        )

        self.out_head3 = nn.Sequential(
            ConvBlock(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            # 输出：[batch, 75, 52, 52]
            nn.Conv2d(in_channels=256, out_channels=self.output_channels, kernel_size=1)
        )

        # YoloLayer层
        anchors = [
            [(116, 90), (156, 198), (373,326)],  # 大尺度 13x13
            [(30, 61), (62, 45), (59, 119)],     # 中尺度 26x26
            [(10,13), (16,30), (33,23)]          # 小尺度 52x52
        ]

        self.yolo_layer = nn.ModuleList([
            YoloLayer(anchors[0], self.num_classes),  # 大尺度 13x13
            YoloLayer(anchors[1], self.num_classes),  # 中尺度 26x26
            YoloLayer(anchors[2], self.num_classes)   # 小尺度 52x52
        ])
    
    def forward(self, x):

        # 获取主干网络的三个特征图输出
        # x52: [batch, 256, 52, 52]
        # x26: [batch, 512, 26, 26]
        # x13: [batch, 1024, 13, 13]
        darknet53_ouput = self.darknet(x)
        x52, x26, x13 = darknet53_ouput
        
        # 第一层的检测路径，输出[batch, 75, 13, 13]
        
        route1 = self.detect_head1(x13)
        detect1_out = self.out_head1(route1)
        # 连接第二层
        route1 = self.route1(route1)
        route1_upsampled = self.upsample1(route1) #[batch, 256, 26, 26]

        # 第二层检测路径
        # 拼接route1_upsampled [batch, 256, 26, 26] + x26 [batch, 512, 26, 26]
        detect2_input = torch.cat([route1_upsampled, x26], dim=1) # 输出：[batch, 768, 26, 26]
        route2 = self.detect_head2(detect2_input) # [batch, 256, 26, 26]
        detect2_out = self.out_head2(route2) # [batch, 75, 26, 26]

        # 连接第三层
        route2 = self.route2(route2)
        route2_upsampled = self.upsample2(route2) # [batch , 128, 52, 52]

        # 第三层检测路径
        # 拼接route2_upsampled [batch, 128, 52, 52] + x52 [batch, 256, 52, 52]
        detect3_input = torch.cat([route2_upsampled, x52], dim = 1) # 输出：[batch, 384, 52, 52]
        detect3_out = self.detect_head3(detect3_input)
        detect3_out = self.out_head3(detect3_out) # 输出：[batch, 75, 52, 52]

        # 通过YoloLayer解析输出
        outputs = []
        for i, layer in enumerate(self.yolo_layer):
            if i == 0:
                output = layer(detect1_out)     # 解析 13x13输出
            elif i == 1:
                output = layer(detect2_out)     # 解析 26x26输出
            else:
                output = layer(detect3_out)     # 解析 52x52输出
            outputs.append(output)

        return outputs

class YoloLayer(nn.Module):
    def __init__(self, anchors, num_classes):
        super().__init__()
        self.anchors = torch.tensor(anchors).float()
        self.num_anchors = len(anchors)
        self.num_classes = num_classes

        # 注册锚点框为缓冲区，不参与梯度更新
        self.register_buffer('anchors_buffer', self.anchors)

    def forward(self, x):
        """
            x的输入shape为：[batch, num_anchors * (5 + num_classes), grid, grid]
            如：[B, 75, 13, 13]
        """
        device = x.device
        batch_size = x.size(0)
        grid_size = x.size(2) 
        self.anchors_buffer = self.anchors_buffer.to(device)


        # 重塑输出维度
        x = x.view(batch_size, self.num_anchors, 5 + self.num_classes, grid_size, grid_size)
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # [batch, num_anchors, grid, grid, num_classes + 5]

        # 解构输出
        # 最后一个维度 5+ num_classes: x, y, w, h, c
        box_xy = torch.sigmoid(x[..., :2]) # 中心点的偏移量
        box_wh = torch.exp(x[..., 2:4]) # 宽高缩放
        confidence = torch.sigmoid(x[..., 4:5]) # 目标置信度
        class_probs = torch.sigmoid(x[..., 5:]) # 类别概率


        # 生成网格坐标
        grid_x, grid_y = torch.meshgrid(
            torch.arange(grid_size, device=device),
            torch.arange(grid_size, device=device),
            indexing='ij'
        )
        grid_x = grid_x.view(1,1,grid_size,grid_size).float()
        grid_y = grid_y.view(1,1,grid_size,grid_size).float()


        # 计算最终边框的坐标
        pred_boxes = torch.zeros_like(x[..., :4]) # 生成和输入一致的shape，xywh均为0的张量
        pred_boxes[..., 0] = (box_xy[..., 0] + grid_x) / grid_size  # x_center
        pred_boxes[..., 1] = (box_xy[..., 1] + grid_y) / grid_size  # y_center

        # anchor_w = self.anchors_buffer[:, 0].view(1, self.num_anchors, 1, 1, 1).to(device)  # [1, num_anchors, 1, 1, 1]
        # anchor_h = self.anchors_buffer[:, 1].view(1, self.num_anchors, 1, 1, 1).to(device)  # [1, num_anchors, 1, 1, 1]
        anchors = self.anchors_buffer.view(1, self.num_anchors, 1, 1, 2).to(device)  # [1, num_anchors, 1, 1, 2]    
        # 计算宽高
        # 这里的box_wh是相对于锚点的缩放
        # 需要将锚点的宽高与网格大小结合起来
        pred_boxes[..., 2] = box_wh[..., 0] * anchors[..., 0] / grid_size # width
        pred_boxes[..., 3] = box_wh[..., 1] * anchors[..., 1] / grid_size # height

        # 组合最终的输出

        output = torch.cat([
            pred_boxes,   # x,y,w,h归一化坐标
            confidence,   # 目标置信度
            class_probs   # 类别概率
        ], dim = -1)

        return output.view(batch_size, self.num_anchors, grid_size, grid_size, 5 + self.num_classes)
