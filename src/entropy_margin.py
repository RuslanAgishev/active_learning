#!/usr/bin/env python

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TORCH_HOME'] = '/home/jovyan/.cache/torch/'
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import *
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
from dataset import CamVid, BDD100K, Cityscapes
from params import *
from datetime import datetime


def al_experiment(model,
                  samples_selection_name,
                  k,
                  experiment_name,
                  visualize_most_uncertain=False,
                  verbose_train=False,
                  random_seed=1):
    """
    Active Learning experiment
    - X_train, y_train: is used partially to train a model
    - X_valid, y_valid: is used fully for validation
    - X_test, y_test: is used as an unlabelled set to detect anomalies and add labels to train set
    """

    # tensorboard logging
    tb = SummaryWriter(log_dir=f'tb_runs/{experiment_name}')
    
    # get the data
    if DATASET_TYPE == CamVid:
        X_train_paths, y_train_paths, X_valid_paths, y_valid_paths = get_camvid_paths(path=os.path.join(DATA_DIR, 'CamVid/'))
    elif DATASET_TYPE == BDD100K:
        X_train_paths, y_train_paths, X_valid_paths, y_valid_paths = get_bdd_paths(path=os.path.join(DATA_DIR, 'bdd100k/seg/'))
    elif DATASET_TYPE == Cityscapes:
        X_train_paths, y_train_paths, X_valid_paths, y_valid_paths = get_cityscapes_paths(path=os.path.join(DATA_DIR, 'Cityscapes/'))
    else:
        print('Choose DATASET_TYPE=CamVid, Cityscapes or BDD100K')

    # define samples selection function from its name
    samples_selection_fn = sample_selection_function(samples_selection_name)
    
    # select k random samples from initial dataset and treat it as initially labelled data
    X_pull = np.copy(X_train_paths)
    y_pull = np.copy(y_train_paths)
    set_random_seed(random_seed)
    initial_selection = np.random.choice(len(X_pull), INITIAL_N_TRAIN_IMAGES, replace=False) # k
    X_train_paths_part = X_pull[initial_selection]
    y_train_paths_part = y_pull[initial_selection]

    X_test = np.delete(X_pull, initial_selection)
    y_test = np.delete(y_pull, initial_selection)

    IoUs = [0.]
    N_train_samples = [0]

    # main loop
    while len(X_train_paths_part) <= MAX_QUEERY_IMAGES:
        # train model
        print('Labelled set size: ', len(X_train_paths_part))
        print('Unlabelled set size: ', len(X_test))
        print(f'\nTraining a model for {model.epochs} epochs...')
        model.train(X_train_paths_part,
                    y_train_paths_part,
                    X_valid_paths,
                    y_valid_paths,
                    Dataset=DATASET_TYPE,
                    verbose=verbose_train)

        # remeber results
        print(f'IoU so far: {model.max_val_iou_score}')
        IoUs.append(model.max_val_iou_score)
        N_train_samples.append(len(X_train_paths_part))
        
        tb.add_scalar('Train IoU vs N train images', model.max_train_iou_score, len(X_train_paths_part))
        tb.add_scalar('Valid IoU vs N train images', model.max_val_iou_score, len(X_train_paths_part))

        if len(X_test) < k:
            print('\nNo more images in Unlabelled set')
            break
            
        selected_images_indexes = samples_selection_fn(X_test, k, model)

        # Visualization
        if visualize_most_uncertain:
            print('Visualizing most uncertain results so far:')
            for i in selected_images_indexes[:1]:
                img_path = X_test[i]
                image = cv2.imread(img_path)[...,(2,1,0)]
                pr_mask = model.predict([img_path])
                mask_np = pr_mask.squeeze().cpu().numpy().round()
                
                plt.figure(figsize=(16, 5))
                title = f'{model.name}_{model.encoder}_N_train_{len(X_train_paths_part)}'
                visualize(image=image, road_mask=mask_np[0,...], car_mask=mask_np[1,...])
                plt.title(title)
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

def define_model(model_name, epochs, batch_size=8, lr=1e-4, seed=0):
    # define model from its name
    set_random_seed(seed)
    model = SegModel(model_name, classes=SEMSEG_CLASSES)
    model.epochs = epochs
    model.batch_size = batch_size
    model.learning_rate = lr
    return model

def main():

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


    results = {}
    dt = datetime.now().strftime("%A_%d_%B_%Y_%I:%M%p") # date time
    # choose model
    for model_name in MODELS:
        print(f'\nModel name: {model_name}')
        print('------------------------------------')
        results[model_name] = {}
        
        # choose samples selection function
        for samples_selection_name in SAMPLES_SELECTIONS:
            print(f'\nSamples selection function: {samples_selection_name}')
            print('------------------------------------')
            results[model_name][samples_selection_name] = {}
            
            # choose number of samples to select for labelling from inference results
            for k, epochs in zip(NUM_UNCERTAIN_IMAGES, MODEL_TRAIN_EPOCHS):
                print(f'\nNumber of samples to label on one iteration, k={k}')
                print('------------------------------------')
                results[model_name][samples_selection_name][str(k)] = {}
                
                experiment_name = f'AL_experiment_{dt}/{model_name}-{samples_selection_name}-{k}'
                model = define_model(model_name, epochs, BATCH_SIZE, INITIAL_LR)
                IoUs, N_train_samples = al_experiment(model,
                                                      samples_selection_name,
                                                      k,
                                                      experiment_name,
                                                      verbose_train=VERBOSE_TRAIN,
                                                      visualize_most_uncertain=VISUALIZE_UNCERTAIN)
                
                results[model_name][samples_selection_name][str(k)]['IoUs'] = IoUs
                results[model_name][samples_selection_name][str(k)]['N_train_samples'] = N_train_samples
                
    pickle_save('./results/'+RESULTS_FNAME, results)

    results = pickle_load('./results/'+RESULTS_FNAME)

    plt.figure(figsize=(8,8))
    # choose model
    for model_name in MODELS:    
        # choose samples selection function
        for samples_selection_name in SAMPLES_SELECTIONS:        
            # choose number of samples to select for labelling from inference results
            for k in NUM_UNCERTAIN_IMAGES:

                ious = results[model_name][samples_selection_name][str(k)]['IoUs']
                n_train = results[model_name][samples_selection_name][str(k)]['N_train_samples']

                plt.plot(np.array(n_train[1:]), ious[1:], label=model_name+'_'+samples_selection_name+'_k='+str(k))
            
    plt.grid()
    plt.title('Active Learning Results', fontsize=18)
    plt.xlabel('N images', fontsize=16)
    plt.ylabel('IoU', fontsize=16)
    plt.legend()
    plt.savefig('results.png')


if __name__=='__main__':
    main()
