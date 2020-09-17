import numpy as np
from tqdm import tqdm


def normalize(x):
    """Scale image to range 0..1 for correct entropy calculation"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x
def entropy(mask_np, axis=0, eps=1e-12):
    mask_np = normalize(mask_np)
    # mask_np.shape = (3, N, M)
    e = np.mean( (-mask_np * np.log2(mask_np+eps)).sum(axis=axis) )
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
    print(f'Min entropy: {np.min(entropies):.3f}, \
            Mean entropy: {np.mean(entropies):.3f}, \
            Max entropy: {np.max(entropies):.3f}')
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
    print(f'Min margin: {np.min(margins):.3f}, \
            Mean margin: {np.mean(margins):.3f}, \
            Max margin: {np.max(margins):.3f}')
    return selected_images_indexes

def random_samples_selection(X, n_samples, model=None):
    selected_images_indexes = np.random.choice(len(X), n_samples, replace=False)
    return selected_images_indexes

def one_hot(mask, axis=0):
    # mask.shape = (c, H, W)
    return (mask == np.max(mask, axis=axis)).astype(np.int)
def vote_entropy(masks, axis=0):
    # masks = [mask_np1, mask_np2, ..., mask_npN]
    # mask_npN.shape = (c, H, W)
    summ = np.zeros_like(masks[0])
    for mask in masks:
        summ += mask
        # summ += one_hot(mask, axis=axis)
    mean = summ / len(masks)
    return entropy(mean, axis=axis)
def vote_entropy_selection(X_test_paths, n_samples, models, iou_threshold=0.4):
    # delete weak models based on IoU iou_threshold
    chosen_models = []
    for model in models:
        if model.max_val_iou_score > iou_threshold:
            chosen_models.append(model)
    if len(chosen_models) == 0:
        print('Models are not trained good enough')
        chosen_models = models
    print(f'Models in ensemble: {len(chosen_models)}')
    # do inference and compute entropy for each image
    vote_entropies = []
    print('Inference on unlabelled data...')
    for img_path in tqdm(X_test_paths):
        masks = [model.predict([img_path]).cpu().numpy().squeeze() for model in chosen_models]
        vote_entropies.append(vote_entropy(masks))
    # Model is mostly uncertain in images with High entropy
    # print('Choosing uncertain images to label...')
    selected_images_indexes = np.argsort(vote_entropies)[::-1][:n_samples]
    print(f'Min Vote entropy: {np.min(vote_entropies):.3f}, \
            Mean Vote entropy: {np.mean(vote_entropies):.3f}, \
            Max Vote entropy: {np.max(vote_entropies):.3f}')
    return selected_images_indexes


def sample_selection_function(name):
    if name=='Random': return random_samples_selection
    elif name=='Entropy': return entropy_selection
    elif name=='Margin': return margin_selection
    elif name=='Committee': return vote_entropy_selection
    else: print('Supported sample selection functions: Random, Entropy, Margin, Committee')