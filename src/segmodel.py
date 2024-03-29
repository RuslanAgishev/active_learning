import torch
import segmentation_models_pytorch as smp
import numpy as np
import cv2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from time import time
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from augmentations import get_preprocessing
from augmentations import get_training_augmentation
from augmentations import get_validation_augmentation
from copy import deepcopy
from params import IMG_HEIGHT, IMG_WIDTH, DATASET_TYPE


class SegModel:
    """Segmentation Models wrapper
    """
    def __init__(self,
                 name='Unet',
                 encoder='mobilenet_v2',
                 encoder_weights='imagenet',
                 classes=['road', 'car']):
        # model params
        self.name = name
        self.arch = self.arch_selection_function(name)
        self.encoder = encoder
        self.encoder_weights = encoder_weights
        self.classes = classes
        self.n_classes = 1 if len(self.classes) == 1 else (len(self.classes) + 1)
        self.activation = 'sigmoid' if len(self.classes) == 1 else 'softmax2d'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.max_train_iou_score = 0
        self.max_val_iou_score = 0
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(self.encoder, self.encoder_weights)
        # training params
        self.learning_rate = 1e-4
        self.weight_decay = 0.5
        self.batch_size = 8
        self.epochs = 1
        self.time0 = time()
        # self.tb = SummaryWriter(log_dir=f'tb_runs/train_{name}_{encoder}_{time()}') # tensorboard

    @staticmethod
    def arch_selection_function(name):
        if name =='Unet':
            arch = smp.Unet
        elif name =='Linknet':
            arch = smp.Linknet
        elif name =='FPN':
            arch = smp.FPN
        elif name =='PSPNet':
            arch = smp.PSPNet
        elif name =='PAN':
            arch = smp.PAN
        elif name =='DeepLabV3':
            arch = smp.DeepLabV3
        else:
            print('Supported sample selection functions: Unet, Linknet, FPN, PSPNet, PAN, DeepLabV3')
            return None
        return arch

    def create_epoch_runners(self,
                             model,
                             loss,
                             metrics,
                             optimizer,
                             verbose=False):
        # it is a simple loop of iterating over dataloader`s samples
        train_epoch = smp.utils.train.TrainEpoch(
            model=model,
            loss=loss, 
            metrics=metrics, 
            optimizer=optimizer,
            device=self.device,
            verbose=verbose,
        )
        valid_epoch = smp.utils.train.ValidEpoch(
            model=model,
            loss=loss, 
            metrics=metrics, 
            device=self.device,
            verbose=verbose,
        )
        return train_epoch, valid_epoch
        
    def create_datasets(self,
                        train_images_paths,
                        train_masks_paths,
                        valid_images_paths,
                        valid_masks_paths,
                        Dataset=DATASET_TYPE):
        train_dataset = Dataset(
            train_images_paths, 
            train_masks_paths, 
            augmentation=get_training_augmentation(), 
            preprocessing=get_preprocessing(self.preprocessing_fn),
            classes=self.classes,
        )
        valid_dataset = Dataset(
            valid_images_paths,
            valid_masks_paths,
            augmentation=get_validation_augmentation(), 
            preprocessing=get_preprocessing(self.preprocessing_fn),
            classes=self.classes,
        )
        return train_dataset, valid_dataset
    
    def create_model(self):
        self.model = self.arch(encoder_name=self.encoder,
                               encoder_weights=self.encoder_weights,
                               classes=self.n_classes,
                               activation=self.activation)
        return self.model

    def train(self,
              train_images_paths,
              train_masks_paths,
              valid_images_paths,
              valid_masks_paths,
              Dataset=DATASET_TYPE,
              verbose=False):
    
        if self.model is None:
            print(f'Creating new model: {self.name}_{self.encoder}')
            self.create_model()
        self.model.train()
        # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
        # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index
        loss = smp.utils.losses.DiceLoss()
        metrics = [
            smp.utils.metrics.IoU(threshold=0.5),
        ]
        optimizer = torch.optim.Adam([ 
            dict(params=self.model.parameters(), lr=self.learning_rate),
        ])
        train_epoch, valid_epoch = self.create_epoch_runners(self.model, loss, metrics, optimizer, verbose=verbose)
        train_dataset, valid_dataset = self.create_datasets(train_images_paths,
                                                            train_masks_paths,
                                                            valid_images_paths,
                                                            valid_masks_paths,
                                                            Dataset=Dataset)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=12)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
        # LR scheduler
        lr_scheduler = StepLR(optimizer, step_size=1, gamma=self.weight_decay)
        # train loop
        train_max_score = 0; val_max_score = 0
        for i in range(0, self.epochs):
            if verbose:
                print('\nEpoch:', i, 'LR:', lr_scheduler.get_last_lr())
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)
            # do something (save model, change lr, etc.)
            lr_scheduler.step() # decay lr
            if val_max_score < valid_logs['iou_score']:
                val_max_score = valid_logs['iou_score']
                torch.save(self.model, f'./{self.name}_{self.encoder}_best_model.pth')
                if verbose: print('Model saved!')
            if train_max_score < train_logs['iou_score']:
                train_max_score = train_logs['iou_score']
            # tensorboard logging
            # self.tb.add_scalar('Valid_Dice_Loss vs Time', valid_logs['dice_loss'], time()-self.time0)
            # self.tb.add_scalar('Train_Dice_Loss vs Time', train_logs['dice_loss'], time()-self.time0)
            # self.tb.add_scalar('Valid_IoU vs Time', valid_logs['iou_score'], time()-self.time0)
            # self.tb.add_scalar('Train_IoU vs Time', train_logs['iou_score'], time()-self.time0)
        # update model with the best saved
        self.max_val_iou_score = val_max_score
        self.max_train_iou_score = train_max_score
        self.model = torch.load(f'./{self.name}_{self.encoder}_best_model.pth').train()

        
    def predict(self, image_paths):
        images = []
        for image_path in image_paths:
            # input preprocessing
            image_raw = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            image = np.copy(image_raw)
            image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
            preprocessing = get_preprocessing(self.preprocessing_fn)
            image = preprocessing(image=image)['image']
            images.append(image)
        # convert to torch tensor and do inference
        x_tensor = torch.tensor(images).to(self.device)
        predictions = self.model.predict(x_tensor)
        return predictions
    
    
