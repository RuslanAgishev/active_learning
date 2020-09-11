import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
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

def find_common_elements(list1, list2):
    """
    >>> list1 = [1,2,3,4,5,6]
    >>> list2 = [3, 5, 7, 9]
    >>> find_common_elements(list1, list2)
    [3, 5]
    """
    return list(set(list1).intersection(list2))

def get_camvid_paths(DATA_DIR='./data/CamVid/'):
    x_train_dir = os.path.join(DATA_DIR, 'train')
    y_train_dir = os.path.join(DATA_DIR, 'trainannot')

    x_valid_dir = os.path.join(DATA_DIR, 'val')
    y_valid_dir = os.path.join(DATA_DIR, 'valannot')

    # x_test_dir = os.path.join(DATA_DIR, 'test')
    # y_test_dir = os.path.join(DATA_DIR, 'testannot')

    X_train_paths = np.array([os.path.join(x_train_dir, image_name) for image_name in os.listdir(x_train_dir)])
    y_train_paths = np.array([os.path.join(y_train_dir, image_name) for image_name in os.listdir(y_train_dir)])

    X_valid_paths = np.array([os.path.join(x_valid_dir, image_name) for image_name in os.listdir(x_valid_dir)])
    y_valid_paths = np.array([os.path.join(y_valid_dir, image_name) for image_name in os.listdir(y_valid_dir)])
    # X_test_paths = np.array([os.path.join(x_test_dir, image_name) for image_name in os.listdir(x_test_dir)])

    return X_train_paths, y_train_paths, X_valid_paths, y_valid_paths


def get_bdd_paths(DATA_DIR='/home/ruslan/datasets/bdd100k/seg/'):
    # BDD100K directories
    x_train_dir = os.path.join(DATA_DIR, 'images/train')
    y_train_dir = os.path.join(DATA_DIR, 'labels/train')

    x_valid_dir = os.path.join(DATA_DIR, 'images/val')
    y_valid_dir = os.path.join(DATA_DIR, 'labels/val')
    # x_test_dir = os.path.join(DATA_DIR, 'images/test')

    # all data paths
    X_train_paths = np.array([os.path.join(x_train_dir, image_name) for image_name in os.listdir(x_train_dir)])
    y_train_paths = np.array([os.path.join(y_train_dir, image_name) for image_name in os.listdir(y_train_dir)])

    X_valid_paths = np.array([os.path.join(x_valid_dir, image_name) for image_name in os.listdir(x_valid_dir)])
    y_valid_paths = np.array([os.path.join(y_valid_dir, image_name) for image_name in os.listdir(y_valid_dir)])
    # X_test_paths = np.array([os.path.join(x_test_dir, image_name) for image_name in os.listdir(x_test_dir)])
    
    return X_train_paths, y_train_paths, X_valid_paths, y_valid_paths