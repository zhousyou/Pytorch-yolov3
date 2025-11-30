import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np

class BasicBlock(nn.Module):
    """Darknet基础残差块"""
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.LeakyReLU(0.1)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels*2)
        self.relu2 = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        
        out += residual
        return out

class Darknet53(nn.Module):
    """Darknet-53主干网络"""
    def __init__(self):
        super(Darknet53, self).__init__()
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.LeakyReLU(0.1)
        
        # 下采样和残差块序列
        self.layer1 = self._make_layer([32, 64], 1)
        self.layer2 = self._make_layer([64, 128], 2)
        self.layer3 = self._make_layer([128, 256], 8)
        self.layer4 = self._make_layer([256, 512], 8)
        self.layer5 = self._make_layer([512, 1024], 4)
        
    def _make_layer(self, channels, num_blocks):
        """构建包含下采样和多个残差块的层"""
        layers = []
        
        # 下采样卷积
        layers.append(("ds_conv", nn.Conv2d(channels[0], channels[1], 
                                          kernel_size=3, stride=2, padding=1, bias=False)))
        layers.append(("ds_bn", nn.BatchNorm2d(channels[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))
        
        # 残差块
        for i in range(num_blocks):
            layers.append((f"residual_{i}", BasicBlock(channels[1], channels[1]//2)))
            
        return nn.Sequential(OrderedDict(layers))
    
    def forward(self, x):
        """前向传播，返回三个尺度的特征图"""
        # 初始卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        # 各层前向传播
        x = self.layer1(x)  # /2
        x = self.layer2(x)  # /4
        
        # 第一个特征图输出 (下采样8倍)
        x = self.layer3(x)  # /8
        out3 = x
        
        # 第二个特征图输出 (下采样16倍)
        x = self.layer4(x)  # /16
        out4 = x
        
        # 第三个特征图输出 (下采样32倍)
        x = self.layer5(x)  # /32
        out5 = x
        
        return out3, out4, out5

class YOLOLayer(nn.Module):
    """YOLO检测层"""
    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.img_dim = img_dim
        
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
        # 网格坐标
        self.grid_size = 0
        self.grid_x = None
        self.grid_y = None
        self.anchor_w = None
        self.anchor_h = None
        
    def forward(self, x, targets=None):
        # x的形状: [batch, anchors*(5+num_classes), grid, grid]
        batch_size = x.size(0)
        grid_size = x.size(2)
        
        # 重新排列输出
        prediction = x.view(batch_size, self.num_anchors, 
                           self.num_classes + 5, grid_size, grid_size)
        prediction = prediction.permute(0, 1, 3, 4, 2).contiguous()
        
        # 获取输出
        x = torch.sigmoid(prediction[..., 0])  # 中心点x
        y = torch.sigmoid(prediction[..., 1])  # 中心点y
        w = prediction[..., 2]  # 宽度
        h = prediction[..., 3]  # 高度
        pred_conf = torch.sigmoid(prediction[..., 4])  # 置信度
        pred_cls = torch.sigmoid(prediction[..., 5:])  # 类别概率
        
        # 如果网格大小改变，重新计算网格
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size)
            
        # 添加网格偏移量并应用锚点尺寸
        pred_boxes = torch.zeros_like(prediction[..., :4])
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h
        
        output = torch.cat([
            pred_boxes.view(batch_size, -1, 4) * self.stride,
            pred_conf.view(batch_size, -1, 1),
            pred_cls.view(batch_size, -1, self.num_classes)
        ], -1)
        
        if targets is None:
            return output, 0
        else:
            # 训练时计算损失
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = self.build_targets(
                pred_boxes, pred_cls, targets, grid_size
            )
            
            loss = self.compute_loss(
                x, y, w, h, pred_conf, pred_cls,
                tx, ty, tw, th, tconf, tcls,
                obj_mask, noobj_mask
            )
            
            return output, loss
            
    def compute_grid_offsets(self, grid_size):
        self.grid_size = grid_size
        self.stride = self.img_dim / self.grid_size
        
        # 网格坐标
        self.grid_x = torch.arange(grid_size).repeat(grid_size, 1).view(
            [1, 1, grid_size, grid_size]
        ).float()
        self.grid_y = torch.arange(grid_size).repeat(grid_size, 1).t().view(
            [1, 1, grid_size, grid_size]
        ).float()
        
        # 锚点尺寸
        self.anchor_w = torch.Tensor(self.anchors).index_select(1, torch.LongTensor([0]))
        self.anchor_h = torch.Tensor(self.anchors).index_select(1, torch.LongTensor([1]))
        self.anchor_w = self.anchor_w.view(1, self.num_anchors, 1, 1).repeat(1, 1, grid_size, grid_size)
        self.anchor_h = self.anchor_h.view(1, self.num_anchors, 1, 1).repeat(1, 1, grid_size, grid_size)
        
    def build_targets(self, pred_boxes, pred_cls, targets, grid_size):
        # 这个函数实现目标匹配，由于篇幅限制，这里简化实现
        # 完整实现需要处理真实框与锚点的匹配
        batch_size = targets.size(0)
        
        # 初始化目标张量
        obj_mask = torch.zeros(batch_size, self.num_anchors, grid_size, grid_size)
        noobj_mask = torch.ones(batch_size, self.num_anchors, grid_size, grid_size)
        tx = torch.zeros(batch_size, self.num_anchors, grid_size, grid_size)
        ty = torch.zeros(batch_size, self.num_anchors, grid_size, grid_size)
        tw = torch.zeros(batch_size, self.num_anchors, grid_size, grid_size)
        th = torch.zeros(batch_size, self.num_anchors, grid_size, grid_size)
        tconf = torch.zeros(batch_size, self.num_anchors, grid_size, grid_size)
        tcls = torch.zeros(batch_size, self.num_anchors, grid_size, grid_size, self.num_classes)
        
        # 简化的目标构建 - 实际实现需要复杂的匹配逻辑
        return None, None, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf
    
    def compute_loss(self, x, y, w, h, pred_conf, pred_cls, tx, ty, tw, th, tconf, tcls, obj_mask, noobj_mask):
        # 损失计算实现
        loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
        loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
        loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
        loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
        loss_conf = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask]) + \
                   0.5 * self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
        loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
        
        total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        return total_loss

class YOLOv3(nn.Module):
    """完整的YOLOv3模型"""
    def __init__(self, num_classes=80, img_dim=416):
        super(YOLOv3, self).__init__()
        
        self.num_classes = num_classes
        self.img_dim = img_dim
        
        # 锚点配置 (COCO数据集)
        self.anchors = [
            [(116, 90), (156, 198), (373, 326)],   # 大尺度
            [(30, 61), (62, 45), (59, 119)],       # 中尺度  
            [(10, 13), (16, 30), (33, 23)]         # 小尺度
        ]
        
        # 主干网络
        self.backbone = Darknet53()
        
        # 检测头网络
        self._build_detection_head()
        
        # YOLO层
        self.yolo_layers = nn.ModuleList([
            YOLOLayer(self.anchors[0], num_classes, img_dim),
            YOLOLayer(self.anchors[1], num_classes, img_dim),
            YOLOLayer(self.anchors[2], num_classes, img_dim)
        ])
        
    def _build_detection_head(self):
        """构建检测头网络"""
        # 检测头1 (大尺度 - 下采样8倍)
        self.head1_conv1 = self._conv_block(1024, 512, 1)
        self.head1_conv2 = self._conv_block(512, 1024, 3)
        self.head1_conv3 = self._conv_block(1024, 512, 1)
        self.head1_conv4 = self._conv_block(512, 1024, 3)
        self.head1_conv5 = self._conv_block(1024, 512, 1)
        self.head1_conv6 = self._conv_block(512, 1024, 3)
        self.head1_out = nn.Conv2d(1024, len(self.anchors[0])*(5+self.num_classes), 1)
        
        # 上采样和连接层1
        self.upsample1 = nn.Sequential(
            self._conv_block(512, 256, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        
        # 检测头2 (中尺度 - 下采样16倍)
        self.head2_conv1 = self._conv_block(768, 256, 1)  # 512+256
        self.head2_conv2 = self._conv_block(256, 512, 3)
        self.head2_conv3 = self._conv_block(512, 256, 1)
        self.head2_conv4 = self._conv_block(256, 512, 3)
        self.head2_conv5 = self._conv_block(512, 256, 1)
        self.head2_conv6 = self._conv_block(256, 512, 3)
        self.head2_out = nn.Conv2d(512, len(self.anchors[1])*(5+self.num_classes), 1)
        
        # 上采样和连接层2
        self.upsample2 = nn.Sequential(
            self._conv_block(256, 128, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        
        # 检测头3 (小尺度 - 下采样32倍)
        self.head3_conv1 = self._conv_block(384, 128, 1)  # 256+128
        self.head3_conv2 = self._conv_block(128, 256, 3)
        self.head3_conv3 = self._conv_block(256, 128, 1)
        self.head3_conv4 = self._conv_block(128, 256, 3)
        self.head3_conv5 = self._conv_block(256, 128, 1)
        self.head3_conv6 = self._conv_block(128, 256, 3)
        self.head3_out = nn.Conv2d(256, len(self.anchors[2])*(5+self.num_classes), 1)
        
    def _conv_block(self, in_channels, out_channels, kernel_size):
        """卷积块: Conv + BatchNorm + LeakyReLU"""
        pad = (kernel_size - 1) // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=pad, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )
        
    def forward(self, x, targets=None):
        # 主干网络提取特征
        small_feat, medium_feat, large_feat = self.backbone(x)
        
        losses = []
        outputs = []
        
        # 检测头1 - 大尺度特征
        x = self.head1_conv1(large_feat)
        x = self.head1_conv2(x)
        x = self.head1_conv3(x)
        x = self.head1_conv4(x)
        route1 = self.head1_conv5(x)  # 用于上采样
        x = self.head1_conv6(route1)
        out1 = self.head1_out(x)
        
        # 第一个YOLO层
        output, loss = self.yolo_layers[0](out1, targets)
        outputs.append(output)
        if loss != 0:
            losses.append(loss)
            
        # 上采样和连接
        x = self.upsample1(route1)
        x = torch.cat([x, medium_feat], 1)
        
        # 检测头2 - 中尺度特征
        x = self.head2_conv1(x)
        x = self.head2_conv2(x)
        x = self.head2_conv3(x)
        x = self.head2_conv4(x)
        route2 = self.head2_conv5(x)  # 用于上采样
        x = self.head2_conv6(route2)
        out2 = self.head2_out(x)
        
        # 第二个YOLO层
        output, loss = self.yolo_layers[1](out2, targets)
        outputs.append(output)
        if loss != 0:
            losses.append(loss)
            
        # 上采样和连接
        x = self.upsample2(route2)
        x = torch.cat([x, small_feat], 1)
        
        # 检测头3 - 小尺度特征
        x = self.head3_conv1(x)
        x = self.head3_conv2(x)
        x = self.head3_conv3(x)
        x = self.head3_conv4(x)
        x = self.head3_conv5(x)
        x = self.head3_conv6(x)
        out3 = self.head3_out(x)
        
        # 第三个YOLO层
        output, loss = self.yolo_layers[2](out3, targets)
        outputs.append(output)
        if loss != 0:
            losses.append(loss)
            
        # 合并所有尺度的输出
        yolo_outputs = torch.cat(outputs, 1)
        
        # 返回结果
        if targets is None:
            return yolo_outputs
        else:
            total_loss = sum(losses)
            return yolo_outputs, total_loss

def load_darknet_weights(model, weights_path):
    """加载预训练的Darknet权重"""
    # 打开权重文件
    with open(weights_path, "rb") as f:
        # 读取头部信息 (主要版本, 次要版本, 子版本, 训练过的图像数量)
        header = np.fromfile(f, dtype=np.int32, count=5)
        # 读取权重
        weights = np.fromfile(f, dtype=np.float32)
    
    ptr = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # 只处理没有偏置的卷积层 (Darknet格式)
            conv_layer = module
            if conv_layer.bias is not None:
                # 加载偏置
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
            # 加载权重
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w
        elif isinstance(module, nn.BatchNorm2d):
            # 加载BatchNorm参数
            bn_layer = module
            num_b = bn_layer.bias.numel()
            bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
            bn_layer.bias.data.copy_(bn_b)
            ptr += num_b
            
            bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
            bn_layer.weight.data.copy_(bn_w)
            ptr += num_b
            
            bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
            bn_layer.running_mean.data.copy_(bn_rm)
            ptr += num_b
            
            bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
            bn_layer.running_var.data.copy_(bn_rv)
            ptr += num_b

# 使用示例
if __name__ == "__main__":
    # 创建模型
    model = YOLOv3(num_classes=80, img_dim=416)
    
    # 打印模型信息
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    x = torch.randn(1, 3, 416, 416)
    output = model(x)
    print(f"输出形状: {output.shape}")
    
    # 测试训练模式
    # 注意: 这里简化了目标格式，实际使用时需要正确的目标格式
    targets = torch.zeros(1, 6)  # [batch, class, x, y, w, h]
    output, loss = model(x, targets)
    print(f"训练输出形状: {output.shape}, 损失: {loss}")