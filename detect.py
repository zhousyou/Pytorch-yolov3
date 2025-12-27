import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import config 
from collections import defaultdict
from torchmetrics.detection.mean_ap import MeanAveragePrecision

class YoloV3Evaluator:
    def __init__(self, model, dataloader, device, num_classes=20, conf_threshold=0.5, nms_threshold=0.5):
        """
        模型评估器

        Args:
            model: 训练好的YOLOv3模型
            dataloader: 验证数据加载器
            device: 计算设备 (CPU或GPU)
            num_classes: 类别数量
            conf_threshold: 置信度阈值
            nms_threshold: NMS阈值
        """
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        # 加载类别名称
        with open(config.voc_classes, 'r') as file:
            self.class_names = [line.strip() for line in file.readlines()]
        
        # 存储所有预测结果和真实标签
        self.eval_results = {
            'predictions': [],
            'ground_truths': [],
            'metrics': defaultdict(list)
        }
    

    def evaluate(self):
        """
            在验证集上进行模型评估
        """

        self.model.eval()

        all_predictions = []
        all_ground_truths = []

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(self.dataloader, desc="Evaluating")):

                # 将数据移动到指定设备
                images = images.to(self.device)

                # 获取模型预测
                predictions = self.model(images) 

                # 处理每张图像的预测结果
                for i in range(images.size(0)):

                    # 获取当前图像的预测结果
                    image_preditions = []
                    for scale_pred in predictions:
                        image_preditions.append(scale_pred[i])  # [anchor, grid, grid, 5+num_classes]
                    
                    # 合并所有尺度的预测
                    combined_predictions = torch.cat(
                        [pred.view(-1, 5 + self.num_classes) for pred in image_preditions],
                        dim=0
                    )

                    # 后处理，应用置信度阈值和NMS
                    detections = self.postprocess(combined_predictions)

                    # 获取对应的真实标签
                    gt_boxes = targets[i]['boxes'].cpu().numpy()
                    gt_labels = targets[i]['labels'].cpu().numpy()
                    
                    # 转换格式用于评估
                    pred_boxes, pred_scores, pred_labels = self.format_detections(detections)

                    all_predictions.append({
                        'boxes': pred_boxes,
                        'scores': pred_scores,
                        'labels': pred_labels,
                        'image_id': batch_idx * self.dataloader.batch_size + i
                    })

                    all_ground_truths.append({
                        'boxes': gt_boxes, 
                        'labels': gt_labels,
                        'image_id': batch_idx * self.dataloader.batch_size + i
                    })

    
    def postprocess(self, predictions):
        """
        后处理： 应用置信度阈值和NMS，返回最终检测结果
        predictions: [nums_predictions, 5 + num_classes]

        输出：
            detections: List of dict, 每个dict包含:
                'bbox': [x1, y1, x2, y2],
                'score': float,
                'class_id': int,
                'class_name': str
        """

        if predictions.size(0) == 0:
            return []
        
        # 分离各个部分
        boxes = predictions[:, :4]  # [x_center, y_center, w, h]归一化坐标
        scores = predictions[:, 4]  # 置信度
        class_probs = predictions[:, 5:]  # 类别概率

        # 找到每个预测的最高类别概率及其索引
        class_scores, class_indices = torch.max(class_probs, dim=1)
        final_scores = scores * class_scores  # 计算最终分数: 置信度 * 类别概率

        # 应用置信度阈值
        mask = final_scores > self.conf_threshold
        boxes = boxes[mask]
        final_scores = final_scores[mask]
        class_indices = class_indices[mask]

        if boxes.size(0) == 0:
            return []
        
        # 转化为[x1, y1, x2, y2]格式
        boxes_corners = torch.zeros_like(boxes)
        boxes_corners[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1  x_center - w/2
        boxes_corners[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1  y_center - h/2
        boxes_corners[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2  x_center + w/2
        boxes_corners[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2  y_center + h/2

        # 应用NMS
        keep_indices = self.nms(boxes_corners, final_scores)

        # 筛选后的检测结果
        filted_boxes = boxes_corners[keep_indices]
        filted_scores = final_scores[keep_indices]
        filted_classes = class_indices[keep_indices]

        # 组合结果
        detections = []
        for i in range(len(filted_boxes)):
            detection = {
                'bbox': filted_boxes[i].cpu().numpy(),
                'score': filted_scores[i].item(),
                'class_id': filted_classes[i].item(),
                'class_name': self.class_names[filted_classes[i].item()]
            }
            detections.append(detection)
        return detections
    
    def nms(self, boxes, scores):
        """
            非极大值抑制 (NMS)

            Args:
                boxes: 形状为 [N, 4] 的边界框张量，格式为 [x1, y1, x2, y2]
                scores: 形状为 [N] 的分数张量
        """

        if boxes.size(0) == 0:
            return []
        
        # 按照置信度排序
        sorted_indices = torch.argsort(scores, descending=True) # 获取排序后的索引,降序
        boxes = boxes[sorted_indices]
        scores = scores[sorted_indices]

        keep = []

        while len(sorted_indices) > 0:
            # 取当前最高置信度的框
            cur_index = sorted_indices[0]
            keep.append(cur_index.item())

            if len(sorted_indices) == 1:
                break

            # 计算当前框与其他框的IOU
            current_box = boxes[0].unsqueeze(0)  # [1,4]
            other_boxes = boxes[1:]  # [N-1,4]

            # 计算IOU
            xx1 = torch.max(current_box[:, 0], other_boxes[:, 0])
            yy1 = torch.max(current_box[:, 1], other_boxes[:, 1])
            xx2 = torch.min(current_box[:, 2], other_boxes[:, 2])
            yy2 = torch.min(current_box[:, 3], other_boxes[:, 3])

            inter_w = torch.clamp(xx2 - xx1, min=0)
            inter_h = torch.clamp(yy2 - yy1, min=0)

            intersectioin = inter_w * inter_h

            area_current = (current_box[:, 2] - current_box[:, 0]) * (current_box[:, 3] - current_box[:, 1])
            area_others = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])

            union = area_current + area_others - intersectioin

            iou = intersectioin / union  # [N-1]

            # 保留IOU小于阈值的框
            keep_indices = torch.where(iou < self.nms_threshold)[0]  # 返回满足条件的索引
            sorted_indices = sorted_indices[keep_indices + 1]  # +1因为第0个是当前框
            boxes = other_boxes[keep_indices] # 更新boxes

        return keep

    def format_detections(self, detections):
        """
            格式化检测结果，便于计算mAP

            输出：
            boxes: np.array, 形状为 [N, 4], 格式为 [x1, y1, x2, y2]
            scores: np.array, 形状为 [N]
            labels: np.array, 形状为 [N]
        """
        boxes = []
        scores = []
        labels = []

        for det in detections:
            boxes.append(det['bbox'])
            scores.append(det['score'])
            labels.append(det['class_id'])
        
        if len(boxes) > 0:
            boxes = np.array(boxes)
            scores = np.array(scores)
            labels = np.array(labels)
            return boxes, scores, labels
        else:
            return np.array([]), np.array([]), np.array([])
    
    def calculate_metrics(self, all_predictions, all_ground_truths):
        """
            计算mAP等评估指标
        """
        
        
