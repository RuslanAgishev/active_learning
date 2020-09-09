import torch
import segmentation_models_pytorch as smp
import numpy as np
import cv2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torch.utils.tensorboard import SummaryWriter
from dataset import Dataset
from augmentations import get_preprocessing, get_training_augmentation, get_validation_augmentation
from time import time

class SegModel:

    def __init__(self, arch=smp.Unet, encoder='resnet18', encoder_weights='imagenet'):
        # model params
        self.arch = arch
        self.encoder = encoder
        self.encoder_weights = encoder_weights
        self.classes = ['sky', 'building', 'pole', 'road', 'pavement', 
                        'tree', 'signsymbol', 'fence', 'car', 
                        'pedestrian', 'bicyclist']
        self.n_classes = 1 if len(self.classes) == 1 else (len(self.classes) + 1)
        self.activation = 'sigmoid' if len(self.classes) == 1 else 'softmax2d'
        self.device = 'cuda' if torch.cuda.is_available else 'cpu'
        self.model = None
        self.max_iou_score = 0
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(self.encoder, self.encoder_weights)
        # training params
        self.learning_rate = 1e-4
        self.batch_size = 8
        self.epochs = 1
        self.time0 = time()
        #self.tb = SummaryWriter(log_dir=f'al_runs/{time()}') # tensorboard

    def create_epoch_runners(self, verbose=False):
        # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
        # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index
        loss = smp.utils.losses.DiceLoss()
        metrics = [
            smp.utils.metrics.IoU(threshold=0.5),
        ]
        optimizer = torch.optim.Adam([ 
            dict(params=self.model.parameters(), lr=self.learning_rate),
        ])
        # create epoch runners 
        # it is a simple loop of iterating over dataloader`s samples
        train_epoch = smp.utils.train.TrainEpoch(
            self.model, 
            loss=loss, 
            metrics=metrics, 
            optimizer=optimizer,
            device=self.device,
            verbose=verbose,
        )
        valid_epoch = smp.utils.train.ValidEpoch(
            self.model, 
            loss=loss, 
            metrics=metrics, 
            device=self.device,
            verbose=verbose,
        )
        return train_epoch, valid_epoch
        
    def create_datasets(self, train_images_paths, train_masks_paths, valid_images_paths, valid_masks_paths):
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
    
    def train(self, train_images_paths, train_masks_paths, valid_images_paths, valid_masks_paths, verbose=False):      
        if self.model is None: self.create_model()
        train_epoch, valid_epoch = self.create_epoch_runners(verbose=verbose)
        train_dataset, valid_dataset = self.create_datasets(train_images_paths,
                                                            train_masks_paths,
                                                            valid_images_paths,
                                                            valid_masks_paths)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=12)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
        
        # train loop
        max_score = 0
        for i in range(0, self.epochs):
            if verbose: print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)
            # do something (save model, change lr, etc.)
            if max_score < valid_logs['iou_score']:
                max_score = valid_logs['iou_score']
                torch.save(self.model, './best_model.pth')
                if verbose: print('Model saved!')
            # tensorboard logging
            #self.tb.add_scalar('Valid_Dice_Loss vs Time', valid_logs['dice_loss'], time()-self.time0)
            #self.tb.add_scalar('Train_Dice_Loss vs Time', train_logs['dice_loss'], time()-self.time0)
            #self.tb.add_scalar('Valid_IoU vs Time', valid_logs['iou_score'], time()-self.time0)
            #self.tb.add_scalar('Train_IoU vs Time', train_logs['iou_score'], time()-self.time0)
                    
        # update model with the best saved
        self.max_iou_score = max_score
        self.model = torch.load('./best_model.pth')
        
    def predict(self, image_paths):
        images = []
        for image_path in image_paths:
            # input preprocessing
            image_raw = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            image = np.copy(image_raw)
            image = cv2.resize(image, (320, 320))
            preprocessing = get_preprocessing(self.preprocessing_fn)
            image = preprocessing(image=image)['image']
            images.append(image)
        # convert to torch tensor and do inference
        x_tensor = torch.tensor(images).to(self.device)
        predictions = self.model.predict(x_tensor)
        return predictions
    
    
def model_selection_function(name):
    if name =='Unet':
        arch = smp.Unet
    elif name =='Linknet':
        arch = smp.Linknet
    elif name =='FPN':
        arch = smp.FPN
    elif name =='PSPNet':
        arch = smp.PSPNet
#     elif name =='PAN':
#         arch = smp.PAN
#     elif name =='DeepLabV3':
#         arch = smp.DeepLabV3
    else:
        print('Supported sample selection functions: Unet, Linknet, FPN, PSPNet')
        return None
    return SegModel(arch=arch)