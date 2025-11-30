import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pycocotools.coco import COCO

class COCODataset(Dataset):
    """
        COCO数据集类

        Args:
            data_dir: 输入数据路径
            annotation_file: 标注文件路径
            img_size: 输入图像尺寸
            transform: 数据增强变换
            is_training: 是否为训练模式
    """

    def __init__(self, data_dir, annotation_file, img_size = 416, transform = None, is_training = True):
        super().__init__()

        self.data_dir = data_dir
        self.annotation_file = annotation_file
        self.img_size = img_size
        self.transform = transform
        self.is_training = is_training

        # 加载COCO标注
        self.coco = COCO(self.annotation_file)
        self.img_ids = self.coco.getImgIds()        # 获取图像的ID

        # 过滤没有标注的图像, 只有标注个数大于0，也就是有标注的情况下才可以保留
        self.image_ids = [img_id for img_id in self.image_ids if len(self.coco.getAnnIds(imgIds=img_id)) > 0]

        # 类别信息
        self.categories = self.coco.loadCats(self.coco.getCatIds())