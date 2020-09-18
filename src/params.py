from dataset import CamVid, BDD100K, Cityscapes


DATASET_TYPE = CamVid # one of CamVid, BDD100K, Cityscapes
DATA_DIR = '/home/ruslan/datasets/'
# DATA_DIR = '/mnt/hard_storage/data/datasets/mseg'
MAX_QUEERY_IMAGES = 140 # 220 # maximum number of images to train on during AL loop
MODEL_TRAIN_EPOCHS = [2, 4] #, 2, 3] # 5 # number of epochs to train a model during one AL cicle
BATCH_SIZE = 16 #! should be less for DeepLab training
INITIAL_LR = 1e-4 # initial learning rate
WEIGHT_DECAY = 0.5 # weight decay rate on every epoch 
INITIAL_N_TRAIN_IMAGES = 20 # 20, initial number of accessible labelled images
NUM_UNCERTAIN_IMAGES = [20, 40] #, 400, 600]#, 400] #, 100] # k: number of uncertain images to label at each AL cicle
SEMSEG_CLASSES = ['road', 'car'] # model output classes
SAMPLES_SELECTIONS = ['Committee', 'Random']
# SAMPLES_SELECTIONS = ['Random', 'Margin', 'Entropy']
MODELS = ['FPN'] # ['Unet', 'Linknet', 'FPN', 'PSPNet'] # for Entropy and Margin samples selection
ENSEMBLE_SIZE = 3 # number of networks in ensemble to vote
VISUALIZE_UNCERTAIN = False # periodically visualize images during active training cycle
VERBOSE_TRAIN = True
IMG_HEIGHT, IMG_WIDTH = 320, 320 # images shape for training and inference



# BDD100K classes:
# ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
# 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
# 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

# CamVid classes:
# ['sky', 'building', 'pole', 'road', 'pavement', 
# 'tree', 'signsymbol', 'fence', 'car', 
# 'pedestrian', 'bicyclist', 'unlabelled']

# Cityscapes classes:
# ['unlabeled', 'ego vehicle', 'rectification border', 'out of roi', 'static', 'dynamic',
# 'ground', 'road', 'sidewalk', 'parking', 'rail track', 'building', 'wall', 'fence',
# 'guard rail', 'bridge', 'tunnel', 'pole', 'polegroup', 'traffic light', 'traffic sign',
# 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'caravan',
# 'trailer', 'train', 'motorcycle', 'bicycle', 'license plate']
