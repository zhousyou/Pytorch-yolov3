import torch
import torch.nn as nn
from pycocotools.coco import COCO
import config
import os

# coco = COCO('person_keypoints_val2017.json')

# catID = coco.getCatIds()
# print(catID)

# cat = coco.loadCats(catID)
# print(cat)

# voc_classes = os.open(config.voc_classes, os.O_RDONLY)
# print(voc_classes)

with open(config.voc_classes, 'r') as file:
    classes = [line.strip() for line in file.readlines()]
print(classes)
print(type(classes), classes[0])
