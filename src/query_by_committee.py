#!/usr/bin/env python

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TORCH_HOME'] = '/home/jovyan/.cache/torch/'
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import visualize
from utils import pickle_load, pickle_save
from utils import get_bdd_paths, get_camvid_paths
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import segmentation_models_pytorch as smp
# segmentation models wrapper
from segmodel import SegModel
# anomaly detection functions
from anomaly_detection import sample_selection_function
# datasets wrapper
from dataset import CamVid, BDD100K
from datetime import datetime


DATASET_TYPE = CamVid
MAX_QUEERY_IMAGES = 120 # 220 # maximum number of images to train on during AL loop
MODEL_TRAIN_EPOCHS = 2 # 5 # number of epochs to train a model during one AL cicle
BATCH_SIZE = 16 #! should be 8 for DeepLab training
INITIAL_LR = 1e-4
INITIAL_N_TRAIN_IMAGES = 20 # 20, initial number of accessible labelled images
NUM_UNCERTAIN_IMAGES = [20] #, 100] # k: number of uncertain images to label at each AL cicle
SEMSEG_CLASSES = ['road', 'car']
SAMPLES_SELECTIONS = ['Random', 'Committee']
ENSEMBLE_SIZE = 3
VISUALIZE_UNCERTAIN = False
VERBOSE_TRAIN = False


def al_experiment(models,
                  k,
                  samples_selection_name='Random',
                  experiment_name='AL_experiment',
                  visualize_most_uncertain=False,
                  verbose_train=False,
                  random_seed=1):

    # get the data
    if DATASET_TYPE == CamVid:
        X_train_paths, y_train_paths, X_valid_paths, y_valid_paths = get_camvid_paths(DATA_DIR='./data/CamVid/')
    elif DATASET_TYPE == BDD100K:
        X_train_paths, y_train_paths, X_valid_paths, y_valid_paths = get_bdd_paths(DATA_DIR='/home/ruslan/datasets/bdd100k/seg/')
    else:
        print('Choose DATASET_TYPE=CamVid or BDD100K')

    # tensorboard
    tb = SummaryWriter(log_dir=f'tb_runs/{experiment_name}')
    
    # choose first data batch to train on
    np.random.seed(random_seed)
    X = np.copy(X_train_paths)
    y = np.copy(y_train_paths)
    
    initial_selection = np.random.choice(len(X), INITIAL_N_TRAIN_IMAGES, replace=False) # k
    X_train_paths_part = X[initial_selection]
    y_train_paths_part = y[initial_selection]

    X_test = np.delete(X, initial_selection)
    y_test = np.delete(X, initial_selection)
    
    samples_selection_fn = sample_selection_function(samples_selection_name)
    
    IoUs = [0.]
    N_train_samples = [0]
    # main AL loop
    while len(X_train_paths_part) <= MAX_QUEERY_IMAGES:
        # train each model in committee
        print('Labelled set size: ', len(X_train_paths_part))
        print('Unlabelled set size: ', len(X_test))
        for model in models:
            print(f'\nTraining a model for {model.epochs} epochs...')
            model.train(X_train_paths_part,
                        y_train_paths_part,
                        X_valid_paths,
                        y_valid_paths,
                        Dataset=CamVid,
                        verbose=verbose_train)
        # remeber results
        val_iou = np.mean([model.max_val_iou_score for model in models])
        max_val_iou = np.max([model.max_val_iou_score for model in models])
        print(f'IoU so far: {val_iou}')
        IoUs.append(val_iou)
        N_train_samples.append(len(X_train_paths_part))

        train_iou = np.mean([model.max_train_iou_score for model in models])
        max_train_iou = np.max([model.max_train_iou_score for model in models])
        tb.add_scalar('Mean Ensemble Train IoU vs N train images', train_iou, len(X_train_paths_part))
        tb.add_scalar('Mean Ensemble Valid IoU vs N train images', val_iou, len(X_train_paths_part))
        tb.add_scalar('Max Ensemble Train IoU vs N train images', max_train_iou, len(X_train_paths_part))
        tb.add_scalar('Max Ensemble Valid IoU vs N train images', max_val_iou, len(X_train_paths_part))
        
        if len(X_test) < k:
            print('\nNo more images in Unlabelled set')
            break

        # select most uncertain samples
        selected_images_indexes = samples_selection_fn(X_test, k, models)

        # Visualization
        if visualize_most_uncertain and samples_selection_name!='Random':
            print('Visualizing most uncertain results so far:')
            for i in selected_images_indexes[:1]:
                img_path = X_test[i]
                image = cv2.imread(img_path)[...,(2,1,0)]
                for model in models:
                    pr_mask = model.predict([img_path])
                    mask_np = pr_mask.squeeze().cpu().numpy().round()
                    
                    plt.figure(figsize=(16, 5))
                    title = f'{model.name}_{model.encoder}_N_train_{len(X_train_paths_part)}'
                    print(title)
                    plt.title(title)
                    visualize(image=image, road_mask=mask_np[0,...], car_mask=mask_np[1,...])
                    plt.show()

        # Add labels for uncertain images to train data
        #print('Labelled set before: ', len(X_train_paths_part))
        X_train_paths_part = np.concatenate([X_train_paths_part, X_test[selected_images_indexes]])
        y_train_paths_part = np.concatenate([y_train_paths_part, y_test[selected_images_indexes]])
        #print('Labelled set after: ', len(X_train_paths_part))

        # Remove labelled data from validation set
        #print('Unlabelled set before: ', len(X_test))
        X_test = np.delete(X_test, selected_images_indexes)
        y_test = np.delete(y_test, selected_images_indexes)
        #print('Unlabelled set after: ', len(X_test))

    print(f'Max IoU score: {np.max(IoUs)}')
    print('----------------------------------------\n')
    return IoUs, N_train_samples


def models_ensemble(n_models=4):
    model1 = SegModel('Unet', encoder='mobilenet_v2', classes=SEMSEG_CLASSES)
    model2 = SegModel('FPN', encoder='mobilenet_v2', classes=SEMSEG_CLASSES)
    model3 = SegModel('PAN', encoder='mobilenet_v2', classes=SEMSEG_CLASSES)
    model4 = SegModel('DeepLabV3', encoder='mobilenet_v2', classes=SEMSEG_CLASSES)
    # choose from defined models
    models = [model1, model2, model3, model4]
    models = models[:n_models]
    for model in models:
        model.epochs = MODEL_TRAIN_EPOCHS
        model.batch_size = BATCH_SIZE
        model.learning_rate = INITIAL_LR
    return models


results = {}
dt = datetime.now().strftime("%A_%d_%B_%Y_%I:%M%p") # date time
# choose samples selection function
for samples_selection_name in SAMPLES_SELECTIONS:
    print(f'\nSamples selection function: {samples_selection_name}')
    print('------------------------------------')
    results[samples_selection_name] = {}
    
    # choose number of samples to select for labelling from inference results
    for k in NUM_UNCERTAIN_IMAGES:
        print(f'\nNumber of samples to label on one iteration, k={k}')
        print('------------------------------------')

        # define models committee
        models = models_ensemble(n_models=ENSEMBLE_SIZE)
            
        results[samples_selection_name][str(k)] = {}

        experiment_name = f'AL_experiment_{dt}/{samples_selection_name}_{k}'
        IoUs, N_train_samples = al_experiment(models,
                                      k,
                                      samples_selection_name,
                                      experiment_name,
                                      visualize_most_uncertain=VISUALIZE_UNCERTAIN,
                                      verbose_train=VERBOSE_TRAIN)
        
        results[samples_selection_name][str(k)]['IoUs'] = IoUs
        results[samples_selection_name][str(k)]['N_train_samples'] = N_train_samples  
pickle_save('results.pkl', results)


results = pickle_load('results.pkl')

plt.figure(figsize=(8,8))
# choose samples selection function
for samples_selection_name in SAMPLES_SELECTIONS:        
    # choose number of samples to select for labelling from inference results
    for k in NUM_UNCERTAIN_IMAGES:

        ious = results[samples_selection_name][str(k)]['IoUs']
        n_train = results[samples_selection_name][str(k)]['N_train_samples']
        
        plt.plot(np.array(n_train[1:]), ious[1:], label=f'{samples_selection_name}_k={k}')
plt.grid()
plt.title('Active Learning Results', fontsize=18)
plt.xlabel('N images', fontsize=16)
plt.ylabel('IoU', fontsize=16)
plt.legend()
plt.savefig('results.png');
