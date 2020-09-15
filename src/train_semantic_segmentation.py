#!/usr/bin/env python

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataset import BDD100K, CamVid
from segmodel import SegModel
from utils import get_bdd_paths, get_camvid_paths


DATASET_TYPE = BDD100K # BDD100K or CamVid

### Load data
if DATASET_TYPE == CamVid:
    X_train_paths, y_train_paths, X_valid_paths, y_valid_paths = get_camvid_paths(DATA_DIR='./data/CamVid/')
elif DATASET_TYPE == BDD100K:
    X_train_paths, y_train_paths, X_valid_paths, y_valid_paths = get_bdd_paths(DATA_DIR='/home/ruslan/datasets/bdd100k/seg/')
else:
    print('Choose DATASET_TYPE="CamVid" or "BDD100K"')

# BDD100K classes:
# ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
# 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
# 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

# CamVid classes:
# ['sky', 'building', 'pole', 'road', 'pavement', 
# 'tree', 'signsymbol', 'fence', 'car', 
# 'pedestrian', 'bicyclist', 'unlabelled']

classes = ['road', 'car']

unet = SegModel('Unet', encoder='resnet34', classes=classes)
unet.epochs = 10
unet.learning_rate = 1e-4
unet.batch_size = 8

unet.train(X_train_paths, y_train_paths, X_valid_paths, y_valid_paths, DATASET_TYPE, verbose=True)

