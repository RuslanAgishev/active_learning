from dataset import CamVid, BDD100K, Cityscapes


DATASET_TYPE = CamVid
MAX_QUEERY_IMAGES = 200 # 220 # maximum number of images to train on during AL loop
MODEL_TRAIN_EPOCHS = [1]#, 2, 3] # 5 # number of epochs to train a model during one AL cicle
BATCH_SIZE = 16 #! should be 8 for DeepLab training
INITIAL_LR = 1e-4
WEIGHT_DECAY = 1
INITIAL_N_TRAIN_IMAGES = 200 # 20, initial number of accessible labelled images
NUM_UNCERTAIN_IMAGES = [200]#, 400, 600]#, 400] #, 100] # k: number of uncertain images to label at each AL cicle
SEMSEG_CLASSES = ['road', 'car']
SAMPLES_SELECTIONS = ['Committee', 'Random'] #['Random', 'Margin', 'Entropy']
MODELS = ['Unet']#, 'Linknet', 'FPN', 'PSPNet']
ENSEMBLE_SIZE = 2
VISUALIZE_UNCERTAIN = False
VERBOSE_TRAIN = True
IMG_HEIGHT, IMG_WIDTH = 320, 320