#!/usr/bin/env python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
import segmentation_models_pytorch as smp
# segmentation models wrapper
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
import pickle

def pickle_save(fname, data):
    filehandler = open(fname,"wb")
    pickle.dump(data,filehandler)
    filehandler.close() 
    print('saved', fname, os.getcwd(), os.listdir())

def pickle_load(fname):
    #print(os.getcwd(), os.listdir())
    file = open(fname,'rb')
    data = pickle.load(file)
    file.close()
    #print(data)
    return data    
    
# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

def get_training_augmentation():
    train_transform = [
        albu.Resize(352, 640, interpolation=1, always_apply=True),

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=352, min_width=640, always_apply=True, border_mode=0),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(352, 640, interpolation=1, always_apply=True) #cv2.INTER_LINEAR
    ]
    return albu.Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def normalize(x):
    """Scale image to range 0..1 for correct entropy calculation"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x
def entropy(mask_np, eps=1e-9):
    mask_np = normalize(mask_np)
    # mask_np.shape = (3, N, M)
    e = np.mean( (-mask_np * np.log2(mask_np+eps)).sum(axis=0) )
    return e
def entropy_selection(X_test_paths, n_samples, model):
    # do inference and compute entropy for each image
    entropies = []
    print('Inference on unlabelled data...')
    for img_path in tqdm(X_test_paths):
        pr_mask = model.predict([img_path])
        mask_np = pr_mask.squeeze().cpu().numpy()
        entropies.append(entropy(mask_np))
    # Model is mostly uncertain in images with High entropy
    #print('Choosing uncertain images to label...')
    selected_images_indexes = np.argsort(entropies)[::-1][:n_samples]
    print(f'Min entropy: {np.min(entropies):.2f}, \
            Mean Entropy: {np.mean(entropies):.2f}, \
            Max entropy: {np.max(entropies):.2f}')
    return selected_images_indexes

def margin(mask_np):
    mask_np = normalize(mask_np)
    rev_probas = np.sort(mask_np, axis=0)[::-1, ...]
    margins_matrix = rev_probas[0,...] - rev_probas[1,...]
    mean_margin = np.mean(margins_matrix)
    return mean_margin
def margin_selection(X_test_paths, n_samples, model):
    # do inference and compute entropy for each image
    margins = []
    print('Inference on unlabelled data...')
    for img_path in tqdm(X_test_paths):
        pr_mask = model.predict([img_path])
        mask_np = pr_mask.squeeze().cpu().numpy()
        margins.append(entropy(mask_np))
    # Model is mostly uncertain in images with Low margin
    #print('Choosing uncertain images to label...')
    selected_images_indexes = np.argsort(margins)[:n_samples]
    print(f'Min margin: {np.min(margins):.2f}, \
            Mean margin: {np.mean(margins):.2f}, \
            Max margin: {np.max(margins):.2f}')
    return selected_images_indexes

def random_samples_selection(X, n_samples, model=None):
    selected_images_indexes = np.random.choice(len(X), n_samples, replace=False)
    return selected_images_indexes

def sample_selection_function(name):
    if name=='Random': return random_samples_selection
    elif name=='Entropy': return entropy_selection
    elif name=='Margin': return margin_selection
    else: print('Supported sample selection functions: Random, Entropy, Margin')


class BDD100K(BaseDataset):
    """Berkeley Deep Drive Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
               'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
               'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'void']
    
    def __init__(
            self, 
            images_paths, 
            masks_paths, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.images_fps = np.sort(images_paths)
        self.masks_fps = np.sort(masks_paths)
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
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

class SegModel:

    def __init__(self, arch=smp.Unet, encoder='resnet18', encoder_weights='imagenet'):
        # model params
        self.arch = arch
        self.encoder = encoder
        self.encoder_weights = encoder_weights
        # CamVid classes
        # self.classes = ['sky', 'building', 'pole', 'road', 'pavement', 
        #                 'tree', 'signsymbol', 'fence', 'car', 
        #                 'pedestrian', 'bicyclist']

        # BDD100K classes
        self.classes = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
                        'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
                        'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'void']
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
        
    def create_datasets(self,
                        train_images_paths,
                        train_masks_paths,
                        valid_images_paths,
                        valid_masks_paths,
                        Dataset=BDD100K):
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
              Dataset=BDD100K,
              verbose=False):
        if self.model is None: self.create_model()
        train_epoch, valid_epoch = self.create_epoch_runners(verbose=verbose)
        train_dataset, valid_dataset = self.create_datasets(train_images_paths,
                                                            train_masks_paths,
                                                            valid_images_paths,
                                                            valid_masks_paths,
                                                            Dataset=Dataset)
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
    else:
        print('Supported sample selection functions: Unet, Linknet, FPN, PSPNet')
        return None
    return SegModel(arch=arch)

def al_experiment(model_str,
                  samples_selection_fn,
                  k,
                  visualize_most_uncertain=False,
                  verbose_train=False,
                  random_seed=0):
    # define model from its name
    model = model_selection_function(model_str)
    model.epochs = MODEL_TRAIN_EPOCHS
    # define samples selection function from its name
    samples_selection_fn = sample_selection_function(samples_selection_str)
    
    # select k random samples from initial dataset and treat it as initially labelled data
    X = np.copy(X_train_paths)
    y = np.copy(y_train_paths)
    np.random.seed(random_seed)
    initial_selection = np.random.choice(len(X), INITIAL_N_TRAIN_IMAGES, replace=False) # k
    X_train_paths_part = X[initial_selection]
    y_train_paths_part = y[initial_selection]

    X_test = np.delete(X, initial_selection)
    y_test = np.delete(X, initial_selection)

    IoUs = [0.]
    N_train_samples = [len(X_train_paths_part)]

    # main loop
    while len(X_train_paths_part) <= MAX_QUEERY_IMAGES:
        # train model
        print('Labelled set size: ', len(X_train_paths_part))
        print('Unlabelled set size: ', len(X_test))
        print(f'\nTraining a model for {MODEL_TRAIN_EPOCHS} epochs...')
        model.train(X_train_paths_part,
                    y_train_paths_part,
                    X_valid_paths,
                    y_valid_paths,
                    Dataset=BDD100K,
                    verbose=verbose_train)

        # remeber results
        print(f'IoU so far: {model.max_iou_score}')
        IoUs.append(model.max_iou_score)
        N_train_samples.append(len(X_train_paths_part))
        
        if len(X_test) < k:
            print('\nNo more images in Unlabelled set')
            break
            
        selected_images_indexes = samples_selection_fn(X_test, k, model)

        # Add labels for uncertain images to train data
        #print('Labelled set before: ', len(X_train_paths_part))
        X_train_paths_part = np.concatenate([X_train_paths_part, X_test[selected_images_indexes]])
        y_train_paths_part = np.concatenate([y_train_paths_part, y_test[selected_images_indexes]])
        #print('Labelled set after: ', len(X_train_paths_part))

        # Visualization
        if visualize_most_uncertain:
            print('Visualizing most uncertain results so far:')
            for i in selected_images_indexes[:1]:
                img_path = X_test[i]
                image = cv2.imread(img_path)[...,(2,1,0)]
                gt_mask = cv2.imread(y_test_paths[i])
                pr_mask = model.predict([img_path])
                mask_np = pr_mask.squeeze().cpu().numpy().round()

                visualize(image=image, car_mask=mask_np[0,...], road_mask=mask_np[1,...])

        # Remove labelled data from validation set
        #print('Unlabelled set before: ', len(X_test))
        X_test = np.delete(X_test, selected_images_indexes)
        y_test = np.delete(y_test, selected_images_indexes)
        #print('Unlabelled set after: ', len(X_test))
        
    print(f'Max IoU score: {np.max(IoUs)}')
    print('----------------------------------------\n')
    return IoUs, N_train_samples


### Load data

# BDD100K directories
# DATA_DIR = '/home/jovyan/bdd100k/seg/'
DATA_DIR = '/home/ruslan/datasets/bdd100k/seg/'
x_train_dir = os.path.join(DATA_DIR, 'images/train')
y_train_dir = os.path.join(DATA_DIR, 'labels/train')

x_valid_dir = os.path.join(DATA_DIR, 'images/val')
y_valid_dir = os.path.join(DATA_DIR, 'labels/val')

x_test_dir = os.path.join(DATA_DIR, 'images/test')

# all data paths
X_train_paths = np.array([os.path.join(x_train_dir, image_name) for image_name in os.listdir(x_train_dir)])
y_train_paths = np.array([os.path.join(y_train_dir, image_name) for image_name in os.listdir(y_train_dir)])

X_valid_paths = np.array([os.path.join(x_valid_dir, image_name) for image_name in os.listdir(x_valid_dir)])
y_valid_paths = np.array([os.path.join(y_valid_dir, image_name) for image_name in os.listdir(y_valid_dir)])

X_test_paths = np.array([os.path.join(x_test_dir, image_name) for image_name in os.listdir(x_test_dir)])


MAX_QUEERY_IMAGES = 2000 # 220 # maximum number of images to train on during AL loop
MODEL_TRAIN_EPOCHS = 5 # 5 # number of epochs to train a model during one AL cicle
INITIAL_N_TRAIN_IMAGES = 1000 # 20, initial number of accessible labelled images
NUM_UNCERTAIN_IMAGES = [500]#, 20]#, 40, 60] # k: number of uncertain images to label at each AL cicle
SAMPLES_SELECTIONS = ['Margin', 'Random']#, 'Entropy']
MODELS = ['Unet']#, 'Linknet', 'FPN', 'PSPNet']


name = ''
for model in MODELS:
    name += model + '_'
name += 'Nsamples_'+str(MAX_QUEERY_IMAGES)
name += '_epochs_'+str(MODEL_TRAIN_EPOCHS)
name += '_N0_'+str(INITIAL_N_TRAIN_IMAGES)
name += '_Ks_'
for k in NUM_UNCERTAIN_IMAGES:
    name += str(k) + '_'
for fn in SAMPLES_SELECTIONS:
    name += fn + '_'
RESULTS_FNAME = name+'.pkl'
print(RESULTS_FNAME)


results = {}

# choose model
for model_str in MODELS:
    print(f'\nModel name: {model_str}')
    print('------------------------------------')
    results[model_str] = {}
    
    # choose samples selection function
    for samples_selection_str in SAMPLES_SELECTIONS:
        print(f'\nSamples selection function: {samples_selection_str}')
        print('------------------------------------')
        results[model_str][samples_selection_str] = {}
        
        # choose number of samples to select for labelling from inference results
        for k in NUM_UNCERTAIN_IMAGES:
            print(f'\nNumber of samples to label on one iteration, k={k}')
            print('------------------------------------')
            results[model_str][samples_selection_str][str(k)] = {}
            
            IoUs, N_train_samples = al_experiment(model_str, samples_selection_str, k, verbose_train=True)
            
            results[model_str][samples_selection_str][str(k)]['IoUs'] = IoUs
            results[model_str][samples_selection_str][str(k)]['N_train_samples'] = N_train_samples
            
pickle_save(RESULTS_FNAME, results)


results = pickle_load(RESULTS_FNAME)
# results = pickle_load('Unet_epochs_2_N0_80_Ks_10_20_Margin_Random_Entropy_.pkl')

plt.figure(figsize=(8,8))

# choose model
for model_str in MODELS:    
    # choose samples selection function
    for samples_selection_str in SAMPLES_SELECTIONS:        
        # choose number of samples to select for labelling from inference results
        for k in NUM_UNCERTAIN_IMAGES:

            ious = results[model_str][samples_selection_str][str(k)]['IoUs']
            n_train = results[model_str][samples_selection_str][str(k)]['N_train_samples']

            plt.plot(np.array(n_train[1:]), ious[1:], label=model_str+'_'+samples_selection_str+'_k='+str(k))
        
plt.grid()
plt.title('Active Learning Results', fontsize=18)
plt.xlabel('N images / full train set size', fontsize=16)
plt.ylabel('IoU', fontsize=16)
plt.legend()
# plt.imsave('al_results.png')
plt.show()
