from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import cv2
import numpy as np


class SemSegDataset(BaseDataset):
    """Base Semantic Segmentation Dataset.
    Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    ALL_CLASSES = []
    
    def __init__(
            self, 
            images_paths, 
            masks_paths, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.images_fps = images_paths
        self.masks_fps = masks_paths
        
        # convert str names to class values on masks
        self.class_values = [self.ALL_CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.images_fps)

class CamVid(SemSegDataset):
    """CamVid Datatset.
    """
    ALL_CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 
                   'tree', 'signsymbol', 'fence', 'car', 
                   'pedestrian', 'bicyclist', 'unlabelled']

class BDD100K(SemSegDataset):
    """Berkeley Deep Drive Dataset.
    """
    ALL_CLASSES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    		       'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    		       'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'void']

class Cityscapes(SemSegDataset):
    """Cityscapes Dataset.
    """
    ALL_CLASSES = ['unlabeled', 'ego vehicle', 'rectification border', 'out of roi', 'static', 'dynamic',
                   'ground', 'road', 'sidewalk', 'parking', 'rail track', 'building', 'wall', 'fence',
                   'guard rail', 'bridge', 'tunnel', 'pole', 'polegroup', 'traffic light', 'traffic sign',
                   'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'caravan',
                   'trailer', 'train', 'motorcycle', 'bicycle', 'license plate']