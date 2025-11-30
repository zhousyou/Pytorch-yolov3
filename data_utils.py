import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import config

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
        self.classes = config.voc_classes

