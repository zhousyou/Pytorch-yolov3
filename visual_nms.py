import numpy as np

def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU
    box1, box2: [x1, y1, x2, y2] 角点坐标
    """
    # 计算交集区域
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # 检查是否有交集
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    # 计算交集面积
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # 计算各自面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算并集面积
    union_area = box1_area + box2_area - inter_area
    
    # 计算IoU
    iou = inter_area / union_area if union_area > 0 else 0.0
    
    return iou

def batch_iou(box, boxes):
    """批量计算一个框与多个框的IoU"""
    # 计算交集
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    
    # 计算交集面积
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)
    intersection = w * h
    
    # 计算各自面积
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # 计算IoU
    union = area_box + area_boxes - intersection
    iou = intersection / (union + 1e-8)  # 加小量防止除零
    
    return iou

def standard_nms(boxes, scores, iou_threshold=0.5):
    """
    标准NMS算法实现
    
    Args:
        boxes: 边界框列表，形状 [N, 4]，格式 [x1, y1, x2, y2]
        scores: 置信度列表，形状 [N]
        iou_threshold: IoU阈值，超过此阈值则视为同一目标
    
    Returns:
        keep_indices: 保留的边界框索引
    """
    if len(boxes) == 0:
        return []
    
    # 1. 按置信度降序排序
    sorted_indices = np.argsort(scores)[::-1]  # 降序
    boxes = boxes[sorted_indices]
    scores = scores[sorted_indices]
    
    # 初始化保留列表
    keep = []
    
    # 2. 循环处理所有框
    while len(boxes) > 0:
        # 取当前置信度最高的框
        current_idx = sorted_indices[0]  # 原始索引
        keep.append(current_idx)
        
        if len(boxes) == 1:
            break
        
        # 3. 计算当前框与剩余框的IoU
        current_box = boxes[0]
        other_boxes = boxes[1:]
        
        # 批量计算IoU
        ious = batch_iou(current_box, other_boxes)
        
        # 4. 保留IoU小于阈值的框
        # 找到IoU小于阈值的框的索引
        keep_indices = np.where(ious < iou_threshold)[0]
        
        # 5. 更新列表，继续处理
        boxes = other_boxes[keep_indices]
        sorted_indices = sorted_indices[keep_indices + 1]  # +1是因为我们移除了第一个
        
    return keep



def visualize_nms_process():
    """可视化NMS的完整过程"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang SC', 'Arial Unicode MS']  # 指定多个字体以备选
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题[citation:1][citation:8]
    
    # 创建示例检测结果
    boxes = np.array([
        [100, 100, 200, 200],    # 框1
        [110, 110, 210, 210],    # 框2，与框1高度重叠
        [120, 120, 220, 180],    # 框3，与框1部分重叠
        [300, 300, 400, 400],    # 框4，完全不同位置
        [105, 105, 205, 205],    # 框5，与框1高度重叠
        [130, 130, 230, 230],    # 框6，与框1部分重叠
    ])
    
    scores = np.array([0.95, 0.93, 0.85, 0.90, 0.92, 0.80])
    
    iou_threshold = 0.5
    
    print("原始检测框（按置信度排序）:")
    for i, (box, score) in enumerate(zip(boxes, scores)):
        print(f"  框{i}: 坐标={box}, 置信度={score:.2f}")
    
    # 应用NMS
    keep_indices = standard_nms(boxes.copy(), scores.copy(), iou_threshold)
    
    print(f"\n应用NMS (IoU阈值={iou_threshold}):")
    print(f"  保留的框索引: {keep_indices}")
    
    # 可视化过程
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. 原始所有框
    axes[0].set_title(f"原始所有检测框")
    for i, (box, score) in enumerate(zip(boxes, scores)):
        color = 'red' if i == np.argmax(scores) else 'blue'
        alpha = 0.3 if i != np.argmax(scores) else 0.5
        rect = patches.Rectangle(
            (box[0], box[1]), box[2]-box[0], box[3]-box[1],
            linewidth=2, edgecolor=color, facecolor=color, alpha=alpha
        )
        axes[0].add_patch(rect)
        axes[0].text(box[0], box[1]-5, f"{score:.2f}", fontsize=10, color='black')
    
    axes[0].set_xlim(0, 500)
    axes[0].set_ylim(0, 500)
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3)
    
    # 2. NMS过程演示
    axes[1].set_title(f"NMS过程: 选择最佳框")
    
    # 找到最佳框（置信度最高）
    best_idx = np.argmax(scores)
    best_box = boxes[best_idx]
    
    # 绘制最佳框
    rect_best = patches.Rectangle(
        (best_box[0], best_box[1]), best_box[2]-best_box[0], best_box[3]-best_box[1],
        linewidth=3, edgecolor='green', facecolor='green', alpha=0.5,
        label=f'最佳框 (score={scores[best_idx]:.2f})'
    )
    axes[1].add_patch(rect_best)
    
    # 计算与最佳框的IoU
    for i, box in enumerate(boxes):
        if i == best_idx:
            continue
        
        iou = calculate_iou(best_box, box)
        color = 'red' if iou > iou_threshold else 'blue'
        label = f'移除 (IoU={iou:.2f})' if iou > iou_threshold else f'保留 (IoU={iou:.2f})'
        
        rect = patches.Rectangle(
            (box[0], box[1]), box[2]-box[0], box[3]-box[1],
            linewidth=2, edgecolor=color, facecolor=color, alpha=0.3,
            label=label if i < 3 else None  # 只显示前几个的标签
        )
        axes[1].add_patch(rect)
        
        # 绘制IoU值
        axes[1].text(box[0], box[1]-5, f"IoU={iou:.2f}", fontsize=8, color='black')
    
    axes[1].set_xlim(0, 500)
    axes[1].set_ylim(0, 500)
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # 3. NMS后结果
    axes[2].set_title(f"NMS后结果 (保留{len(keep_indices)}个框)")
    
    for i, idx in enumerate(keep_indices):
        box = boxes[idx]
        score = scores[idx]
        
        color = 'green' if i == 0 else 'blue'
        rect = patches.Rectangle(
            (box[0], box[1]), box[2]-box[0], box[3]-box[1],
            linewidth=2, edgecolor=color, facecolor=color, alpha=0.5,
            label=f'框{idx}: score={score:.2f}'
        )
        axes[2].add_patch(rect)
        
        axes[2].text(box[0], box[1]-5, f"{score:.2f}", fontsize=10, color='black')
    
    axes[2].set_xlim(0, 500)
    axes[2].set_ylim(0, 500)
    axes[2].invert_yaxis()
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    return keep_indices

keep_indices = visualize_nms_process()