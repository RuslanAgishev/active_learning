#!/usr/bin/env python

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataset import BDD100K, CamVid, Cityscapes
from segmodel import SegModel
from utils import get_bdd_paths, get_camvid_paths, get_cityscapes_paths
from params import DATA_DIR


DATASET_TYPE = CamVid # BDD100K, Cityscapes or CamVid

### Load data
if DATASET_TYPE == CamVid:
    X_train_paths, y_train_paths, X_valid_paths, y_valid_paths = get_camvid_paths(path=os.path.join(DATA_DIR, 'CamVid/'))
elif DATASET_TYPE == BDD100K:
    X_train_paths, y_train_paths, X_valid_paths, y_valid_paths = get_bdd_paths(path=os.path.join(DATA_DIR, 'bdd100k/seg/'))
elif DATASET_TYPE == Cityscapes:
    X_train_paths, y_train_paths, X_valid_paths, y_valid_paths = get_cityscapes_paths(path=os.path.join(DATA_DIR, 'Cityscapes/'))
else:
    print('Choose DATASET_TYPE=CamVid, Cityscapes or BDD100K')

# BDD100K classes:
# ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
# 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
# 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

# CamVid classes:
# ['sky', 'building', 'pole', 'road', 'pavement', 
# 'tree', 'signsymbol', 'fence', 'car', 
# 'pedestrian', 'bicyclist', 'unlabelled']

classes = ['road', 'car']

model_names = ['Unet']#, 'Linknet', 'FPN', 'PSPNet', 'PAN', 'DeepLabV3']
ious = []

for model_name in model_names:
    model = SegModel(model_name, encoder='mobilenet_v2', classes=classes)
    model.epochs = 1
    model.learning_rate = 1e-4
    model.batch_size = 8

    model.train(X_train_paths, y_train_paths, X_valid_paths, y_valid_paths, DATASET_TYPE, verbose=True)
    iou = model.max_val_iou_score
    ious.append(iou)
    print(f'\n{model_name}_{model.encoder} IoU score: {iou}')

print('\nResults:')
for i in range(len(model_names)):
    print(f'\n{model_names[i]} IoU score: {ious[i]}')
