import cv2
import torch
import config
import colorsys
from torchvision.ops import nms
import numpy as np 
import matplotlib.pyplot as plt
from model.yolo import YoloV3

class YoloV3Predictor:
    def __init__(self, model, device, img_size = 416, conf_threshold = 0.5, nms_threshold = 0.5):
        """
        YOLOv3预测器

        Args:
        
            :param model: 训练好的模型
            :param device: 计算设备
            :param img_size: 输入图像尺寸
            :param conf_threshold: 置信度阈值
            :param nms_threshold: NMS阈值
        """

        self.model = model
        self.device = device
        self.img_size = img_size
        self.conf_thershold = conf_threshold
        self.nms_threshold = nms_threshold

        with open(config.voc_classes , 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        
        # 设置每个类别的颜色
        self.colors = self.generate_colors(self.num_classes)

    
    @property
    def num_classes(self):
        return len(self.class_names)
    
    def generate_colors(self, num_colors):
        colors = []
        for i in range(num_colors):
            hue = i/num_colors
            lightness = 0.5
            saturation = 0.8
            rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
            colors.append([int(rgb[0]*255),int(rgb[1]*255),int(rgb[2]*255)])
        return colors
    
    def preprocess_images(self, img_path):
        """
        预处理图像
        
        Args:
    
            :param img_path: 输入图像的路径
        """

        # 读取图像
        img = cv2.imread(img_path)

        # 保留原始图像的尺寸
        original_image = img.copy()
        original_w, original_h = img.shape[:2]

        # 转换颜色空间 bgr2rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 调整到适合输入模型的大小
        resized_img = cv2.resize(img, (self.img_size, self.img_size))

        # 归一化并转换为张量
        img_tensor = torch.from_numpy(resized_img).permute(2,0,1).float()/255.
        img_tensor = img_tensor.unsqueeze(0)

        # 返回值 ：原始图像，rgb图像，调整大小图像，图像张量，原始尺寸
        return {
            'original_image':original_image,
            'rgb_image':img,
            'resized_image':resized_img,
            'tensor_image':img_tensor,
            'original_shape':(original_w, original_h)
        }
    
    def predict(self, img_path):
        """
        对单张图片进行预测
        
        :param img_path: 输入图片的路径
        """

        # 预处理图像
        preprocessed_img = self.preprocess_images(img_path)

        # 移动图像到计算设备
        img_tensor = preprocessed_img['tensor_image'].to(self.device)

        # 模型推理
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(img_tensor)

        # 对模型输出进行后处理
        detections = []

        # 处理每个尺度的预测
        for scale_pred in predictions:
            scale_pred = scale_pred.view(-1, 5 + self.num_classes)
            boxes = scale_pred[:,:4]
            scores = scale_pred[:, 4]
            class_probs = scale_pred[:, 5:]

            # 找到每个预测最可能的类别
            class_scores, class_indices = torch.max(class_probs, dim=1)

            # 综合置信度，得到最终的得分
            final_scores = class_scores * scores

            # 应用置信度的阈值
            mask = final_scores > self.conf_thershold
            boxes = boxes[mask]
            scores = scores[mask]
            class_probs = class_probs[mask]

            if len(boxes) == 0:
                continue

            # 转换为点坐标
            box_xy = torch.zeros_like(boxes)
            box_xy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2    # x1
            box_xy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2    # y1
            box_xy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2    # x2
            box_xy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2    # y2


            # 转换到原始图像的尺寸
            scale_w, scale_h = preprocessed_img['original_shape']
            box_xy[:, 0] *= scale_w
            box_xy[:, 1] *= scale_h
            box_xy[:, 2] *= scale_w
            box_xy[:, 3] *= scale_h

            # 将结果添加到检测列表
            for i in range(len(box_xy)):
                detection = {
                    'bbox': box_xy[i].cpu().numpy(),
                    'score': scores[i].item(),
                    'class_id': class_indices[i].item(),
                    'class_name': self.class_names[class_indices[i].item()]
                }
                detections.append(detection)
        
        # NMS 
        if detections:
            detections = self.apply_nums(detections)
        
        return {
            'image_info': preprocessed_img,
            'detections': detections,
            'num_detections': len(detections)
        }

    
    def apply_nums(self, detections):
        """
        应用非极大值抑制
        
        Args:
            detections: 输入检测列表
            每个元素是一个字典：{'bbox','score','class_id','class_name'}
        """
        if not detections:
            return []

        # 提取边界框和分数 boxes, scores, classes
        boxes = np.array([det['bbox'] for det in detections])     
        scores = np.array([det['score'] for det in detections])   
        classes = np.array([det['class_id'] for det in detections])

        # 按照类别分别应用nms 
        unique_classes = np.unique(classes) # 
        keep = []


        for class_id in unique_classes:
            # 获取当前类别的检测
            class_mask = classes == class_id
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]

            if len(class_boxes) == 0:
                continue

            # 转换成torch张量，进行NMS检测
            class_boxes_tensor = torch.tensor(class_boxes, device=self.device)
            class_scores_tensor = torch.tensor(class_scores, device=self.device)

            # 应用NMS
            keep_indices = nms(class_boxes_tensor, class_scores_tensor, self.nms_threshold)


            # 添加保留的检测
            for idx in keep_indices.cpu().numpy():
                original_idx = np.where(class_mask)[0][idx]
                keep.append(detections[original_idx])
        
        return keep
    
    def visualize_detections(self, img_path, save_path = None, show_labels = True, show_scores = True, thickness = 2, font_scale = 0.5):
        """
        可视化检测结果
        
        :param img_path: 说明
        :param save_path: 说明
        :param show_labels: 说明
        :param show_scores: 说明
        :param thickness: 说明
        :param font_scale: 说明
        """

        # 获取预测的结果
        result = self.predict(img_path)

        # 获取原始图像 + 检测结果
        original_img = result['image_info']['rgb_image']
        detections = result['detections']

        # 创建绘图副本
        image_with_boxes = original_img.copy()

        # 绘制边界框
        for det in detections:
            boxes = det['bbox'].astype(int)
            scores = det['score']
            class_id = det['class_id']
            class_name = det['class_name']

            print(class_name)

            # 获取颜色
            color = self.colors[class_id]

            # 画框
            cv2.rectangle(image_with_boxes,
                          (boxes[0], boxes[1]),
                          (boxes[2], boxes[3]),
                          color=color,
                          thickness=thickness)
            
        
        # 显示结果
        fig, axes = plt.subplots(1, 2, figsize=(8,16))

        # 原始图像
        axes[0].imshow(original_img)
        axes[0].set_title('Original image')
        axes[0].axis('off')

        # 检测后图像
        axes[1].imshow(image_with_boxes)
        axes[1].set_title(f'Detection ({len(detections)} objects)')
        axes[1].axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = YoloV3(20).to(device)
    checkpoint = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    yolopredict = YoloV3Predictor(model, device)
    img_path = "/root/Pytorch-yolov3/test_img/000005.jpg"
    yolopredict.visualize_detections(img_path)

