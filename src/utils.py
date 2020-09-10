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