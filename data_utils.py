import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import config
import os
import albumentations as A 
from albumentations.pytorch import ToTensorV2

class VOCDataset(Dataset):
    def __init__(self, data_dir, split = 'train', img_size = 416, transform = None):
        """
            VOC数据集类

            Args:
                data_dir: VOC数据集目录
                split: 数据集划分'train','val','trainval'
                img_size:输入图像尺寸
                transform：数据增强变换
        """
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.transform = transform
        
        # VOC数据集的20个类别
        with open(config.voc_classes, 'r') as file:
            self.classes = [line.strip() for line in file.readlines()]
        
        # 建立类别-索引映射字典
        self.class_to_id = {name: idx for idx, name in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        # 加载图像列表
        self.image_ids = self._load_image_ids()

        # 打印加载图像信息
        print(f"加载VOC2007{self.split}数据集:{len(self.image_ids)}张图像,{self.num_classes}个种类")

    
    def _load_image_ids(self):
        """
            加载图像ID列表,返回train/val/trainval.txt中对应图像id
        """

        split_file = os.path.join(
            self.data_dir, 'ImageSets', 'Main', f'{self.split}.txt'
        )
        with open(split_file, 'r') as file:
            image_ids = [line.strip() for line in file.readlines()]
        
        return image_ids
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, index):

        img_id = self.image_ids[index]
        # --------------------------
        # 加载图像
        # --------------------------
        img_path = os.path.join(
            self.data_dir, 'JPEGImages', f'{img_id}.jpg'
        )
        img = cv2.imread(img_path)

        # 异常处理
        if img is None:
            raise FileNotFoundError(f"无法加载图像: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        original_h, original_w = img.shape[:2] # 图像的格式是[B, C, H, W]

        # --------------------------
        # 加载标注
        # --------------------------
        annotation_path = os.path.join(
            self.data_dir, 'Annotations', f'{img_id}.xml'
        )
        boxes, labels = self._parse_annotation(annotation_path, original_w, original_h)

        if len(boxes) == 0:
            # 如果没有标注返回空
            boxes = torch.zeros((0,4))
            labels = torch.zeros((0,))
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)


        # ------------------------------
        # 应用数据增强
        # ------------------------------

        if self.transform:
            # 将边界框转换为[xmin, ymin, xmax, ymax]的格式
            albumentations_boxes = []
            for box in boxes:
                x_center, y_center, w, h = box
                x_min = (x_center - w/2) * original_w
                x_max = (w/2 + x_center) * original_w
                y_min = (y_center - h/2) * original_h
                y_max = (y_center + h/2) * original_h
                albumentations_boxes.append([x_min, y_min, x_max, y_max])

            # 应用变换
            transformed = self.transform(
                img = img,
                bboxes = albumentations_boxes,
                class_labels = labels.tolist()
            )
            image = transformed['img']
            if len(transformed['bboxes']) > 0:
                # 转换回YOLO格式
                new_boxes = []
                for bbox in transformed['bboxes']:
                    x_min, y_min, x_max, y_max = bbox
                    x_center = (x_min + x_max) / 2 / self.img_size
                    y_center = (y_min + y_max) /2 / self.img_size
                    width = (x_max - x_min) / self.img_size
                    height = (y_max - y_min) / self.img_size
                    new_boxes.append([x_center, y_center, width, height])

                boxes = torch.tensor(new_boxes, dtype=torch.float32)
                labels = torch.tensor(transformed['class_labels'], dtype=torch.long)
            else:
                boxes = torch.zeros((0,4))
                labels = torch.zeros((0,))
        else:
            # 调整大小以及归一化
            image = cv2.resize(image, (self.img_size, self.img_size))
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)

        # 构建目标张量

        target_tensor = {
            'boxes':boxes,
            'labels':labels,
            'image_ids':torch.tensor([index]),
            'original_size':torch.tensor([original_h, original_w])
        }

        return image, target_tensor



    def _parse_annotation(self, annotation_path, original_w, original_h):
        """
            解析xml标注文件
            Args:
            * annotation_path : xml标注文件的路径
            * original_w: 原始图片的w
            * original_h: 原始图片的h

            输出：
            * boxes: 一个二维列表，包含xml文件中所有标注框的位置信息
            * labels: 表示xml文件中物体所对应的类别ID
        """

        tree = ET.parse(annotation_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall('object'):
            # 获取xml文件中的每个物体类别
            class_name = obj.find('name').text
            if class_name not in self.class_to_id:
                continue

            # 获取xml中边界框的信息
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            # 转换成模型中的[x, y, w, h]的格式，并归一化
            x_center = ((xmin + xmax) / 2) / original_w
            y_center = ((ymin + ymax) / 2) / original_h
            width = (xmax - xmin) / original_w
            height = (ymax - ymin) / original_h

            if (x_center >=0 and x_center <=1 and y_center >= 0 and y_center <= 1 and width > 0 and width <= 1 and height > 0 and height <= 1):
                boxes.append([x_center, y_center, width, height])
                labels.append(self.class_to_id[class_name])
        return boxes, labels

def get_voc_transforms(img_size = 416, is_training = True):
    """
        获取VOC数据增强变换
        Args:
        * img_size: 要转换成的图片尺寸
        * is_training:是否是训练模式
    """
    if is_training:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p = 0.5),      # 以50%的概率进行水平翻转
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5), #随机调整图片的亮度，对比度，饱和度，色调
            A.Blur(blur_limit=3, p = 0.1),  # 模糊图片
            A.ToGray(p=0.1),      # 转化为灰度
            A.Normalize(mean=[0,0,0], std=[1,1,1]), # 归一化
            ToTensorV2(),   #转化为pytorch可以处理的tensor
        ], bbox_params=A.BboxParams(format='pascal_voc' ,label_fields=['class_labels']))
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0,0,0], std=[1,1,1]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

def create_voc_dataloaders(data_dir, batch_size = 8, img_size = 416, num_workers = 4):
    """
        创建VOC数据加载器
        Args:
        * data_dir: 输入数据的路径
        * batch_size: 数据的批次大小
        * img_size: 默认图片的大小
        * num_workers
    """
    # train_datasets = []
    # val_datasets = []

    # 训练集
    train_transform = get_voc_transforms(img_size=img_size, is_training=True)
    train_dataset = VOCDataset(
        data_dir=data_dir,
        split='train',
        img_size=img_size,
        transform=train_transform
    )
    # train_datasets.append(train_dataset)

    # 验证集
    val_transform = get_voc_transforms(img_size=img_size, is_training=False)
    val_dataset = VOCDataset(
        data_dir=data_dir,
        split='val',
        img_size=img_size,
        transform=val_transform
    )
    # val_datasets.append(val_dataset)

    # 创建数据加载器
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    print(f"训练集：{len(train_dataset)}张图像")
    print(f"验证集：{len(val_dataset)}张图像")

    return train_dataloader, val_dataloader


def collate_fn(batch):
    """
        自定义批次整理函数
    """
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    images = torch.stack(images)
    return images, targets

if __name__ == "__main__":
    data_dir = 'VOCdevkit/VOC2007'
    VOCDataset(data_dir=data_dir,
               split='train')
    train, val = create_voc_dataloaders(data_dir=data_dir)