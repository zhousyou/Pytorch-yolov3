import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import config
import os

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
    
    

if __name__ == "__main__":
    data_dir = 'VOCdevkit/VOC2007'
    VOCDataset(data_dir=data_dir,
               split='train')