#!/usr/bin/env python

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataset import BDD100K
from segmodel import SegModel


DATA_DIR = '/home/ruslan/datasets/bdd100k/seg/'
x_train_dir = os.path.join(DATA_DIR, 'images/train')
y_train_dir = os.path.join(DATA_DIR, 'labels/train')

x_valid_dir = os.path.join(DATA_DIR, 'images/val')
y_valid_dir = os.path.join(DATA_DIR, 'labels/val')

x_test_dir = os.path.join(DATA_DIR, 'images/test')

# all data paths
X_train_paths = np.sort([os.path.join(x_train_dir, image_name) for image_name in os.listdir(x_train_dir)])
y_train_paths = np.sort([os.path.join(y_train_dir, image_name) for image_name in os.listdir(y_train_dir)])

X_valid_paths = np.sort([os.path.join(x_valid_dir, image_name) for image_name in os.listdir(x_valid_dir)])
y_valid_paths = np.sort([os.path.join(y_valid_dir, image_name) for image_name in os.listdir(y_valid_dir)])

X_test_paths = np.sort([os.path.join(x_test_dir, image_name) for image_name in os.listdir(x_test_dir)])


classes = ['road', 'sidewalk', 'building']#, 'wall', 'fence', 'pole', 'traffic light',
           # 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
           # 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
unet = SegModel('Unet', classes=classes)
unet.epochs = 1
unet.train(X_train_paths, y_train_paths, X_valid_paths, y_valid_paths, BDD100K, verbose=True)

