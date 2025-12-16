import torch
import torch.nn as nn
import config

class YOLOLoss(nn.Module):
    def __init__(self, anchors, img_size = 416):
        super().__init__()
        self.num_classes = config.num_classes   # 类别为20
        self.anchors = anchors
        self.num_anchors = len(self.anchors[0])
        self.img_size = img_size
        self.ignore_thershold = 0.5

        # 损失权重
        self.lambda_coord = 5.0     # 坐标损失的权重
        self.lambda_obj = 1.0       # 有目标置信度损失权重
        self.lambda_noobj = 0.5     # 无目标置信度损失权重
        self.lambda_class = 1.0     # 类别损失权重

        self.mse_loss = nn.MSELoss(reduction='mean')    # 均方误差 用于坐标
        self.bce_loss = nn.BCELoss(reduction='mean')    # 二元交叉熵误差用于类别置信度

    def forward(self, predictions, targets):
        """
            Args:
            * predictions:三个尺度预测的输出结果:[scale1, scale2, scale3]
            * targets: 目标列表，每个元素是包含'boxes'和'labels'的字典
        """
        target_loss = 0
        coord_loss = 0
        obj_loss = 0
        noobj_loss = 0
        class_loss = 0

        for scale_idx, pred in enumerate(predictions):
            # print(f"Debug Info: pred shape at scale {scale_idx}: {pred.shape}")
            batch_size = pred.size(0)    # 获取当前输出的批次
            grid_size = pred.size(2)     # 获取输出的尺度
            stride = self.img_size / grid_size  # 计算当前尺度的步幅（32，16，8）

            # 获取当前尺度的锚点框
            scale_anchors = torch.tensor(self.anchors[scale_idx], device=pred.device).float()/stride


            # 构建目标张量
            targer_tensor = self.build_targets(pred, targets, scale_anchors, grid_size, scale_idx)

            # 解析预测
            pred_boxes = pred[..., :4]  # x,y,w,h
            pred_obj = pred[..., 4]   # 目标置信度
            pred_class = pred[..., 5:]  # 类别概率

            # 解析目标
            target_boxes = targer_tensor[..., :4]   # x,y,w,h
            target_obj = targer_tensor[..., 4]  # 置信度
            target_class = targer_tensor[..., 5:]   #类别概率
            # print(f"Debug Info: target_obj shape at scale {scale_idx}: {target_obj.shape}")
            # 计算掩码  ?
            obj_mask = target_obj == 1
            noobj_mask = target_obj == 0

            # # 调试信息
            # print(f"调试信息 - Scale {scale_idx}:")
            # print(f"  Pred Boxes Shape: {pred_boxes.shape}")
            # print(f" obj_mask shape: {obj_mask.shape}, obj_mask sum: {obj_mask.sum().item()}")

            # # 检查索引操作的结果
            # if obj_mask.sum() > 0:
            #     indexed_pred_boxes = pred_boxes[obj_mask]
            #     indexed_target_boxes = target_boxes[obj_mask]
            #     print(f"  Indexed Pred Boxes Shape: {indexed_pred_boxes.shape}")
            #     print(f"  Indexed Target Boxes Shape: {indexed_target_boxes.shape}")

            # 计算坐标的损失
            if obj_mask.sum() > 0:
                # 边界框的坐标损失

                coord_loss_scale = self.mse_loss(
                    pred_boxes[obj_mask][:, :2],
                    target_boxes[obj_mask][:, :2]
                ) + self.mse_loss(
                    torch.sqrt(pred_boxes[obj_mask][:, 2:4]),
                    torch.sqrt(target_boxes[obj_mask][:, 2:4])
                )
                coord_loss += coord_loss_scale
            
            # 目标置信度的损失
            obj_loss_scale = self.bce_loss(
                pred_obj[obj_mask],
                target_obj[obj_mask]
            )
            obj_loss += obj_loss_scale

            # 无目标置信度的损失
            noobj_loss_scale = self.bce_loss(
                pred_obj[noobj_mask],
                target_obj[noobj_mask]
            )
            noobj_loss += noobj_loss_scale

            # 类别损失
            if obj_mask.sum() > 0:
                class_loss_scale = self.bce_loss(
                    pred_class[obj_mask],
                    target_class[obj_mask]
                )
                class_loss += class_loss_scale
        
        # 计算加权总损失
        total_loss = self.lambda_coord *  coord_loss + self.lambda_obj * obj_loss + self.lambda_noobj * noobj_loss + self.lambda_class * class_loss

        # 归一化
        total_loss /= batch_size

        return total_loss, {
            'total_loss': total_loss.item(),
            'coord_loss': coord_loss.item() / batch_size,
            'obj_loss': obj_loss.item() / batch_size,
            'noobj_loss': noobj_loss.item() / batch_size,
            'class_loss': class_loss.item() / batch_size
        }
    
    def build_targets(self, predictions, targets, scaled_anchors, grid_size, scale_idx):
        """
            构建目标张量
        """
        batch_size = predictions.size(0)
        num_anchors = len(scaled_anchors)
        
        # 初始化目标张量
        target_tensor = torch.zeros(
            batch_size, num_anchors, grid_size, grid_size, 5 + self.num_classes,
            device=predictions.device
        )
        
        
        # # 将锚点框缩放到当前网格尺度
        # scale_anchors = torch.tensor(anchors, device=predictions.device).float() / (self.img_size * self.img_size)

        # 遍历批次中的每个图像
        for batch_idx in range(batch_size):
            target = targets[batch_idx]
            boxes = target['boxes']
            labels = target['labels']

            if len(boxes) == 0:
                continue
            
            # gxy = boxes[:, :2] * grid_size  # 计算中心点在网格上的位置
            # gwh = boxes[:, 2:] * grid_size  # 计算宽高在网格上的大小

            # 遍历每个真实边界框
            # 为每个目标找到匹配的锚点框
            for box_idx, (box, label) in enumerate(zip(boxes, labels)):
                # 转换边界框格式
                x_center, y_center, width, height = box

                # 计算网格位置
                grid_x = int(x_center * grid_size)
                grid_y = int(y_center * grid_size)

                if grid_x >= grid_size or grid_y >= grid_size:
                    continue

                # 计算与锚点框的IOU
                ious = []
                for anchor_idx, anchor in enumerate(scaled_anchors):
                    anchor_w, anchor_h = anchor
                    # 计算IOU
                    inter_w = min(width * grid_size, anchor_w)
                    inter_h = min(height * grid_size, anchor_h)
                    inter_area = inter_w * inter_h
                    union_area = anchor_w * anchor_h + width * grid_size + height * grid_size - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0
                    ious.append(iou)

                # 根据IOU选择最优的锚点框（IOU最大），返回最优框的索引
                best_anchor = torch.argmax(torch.tensor(ious))

                # 设置目标值. [batch, num_anchors, gridsize, gridsize, 5+numclasses]
                target_tensor[batch_idx, best_anchor, grid_y, grid_x, 0] = x_center * grid_size - grid_x    # 中心点x的偏移量
                target_tensor[batch_idx, best_anchor, grid_y, grid_x, 1] = y_center * grid_size - grid_y    # 中心点y的偏移量
                target_tensor[batch_idx, best_anchor, grid_y, grid_x, 2] = width * grid_size / scaled_anchors[best_anchor][0]    #w
                target_tensor[batch_idx, best_anchor, grid_y, grid_x, 3] = height * grid_size / scaled_anchors[best_anchor][1]    #h
                target_tensor[batch_idx, best_anchor, grid_y, grid_x, 4] = 1    # 置信度
                target_tensor[batch_idx, best_anchor, grid_y, grid_x, 5 + label] = 1    #类别概率

        return target_tensor              





