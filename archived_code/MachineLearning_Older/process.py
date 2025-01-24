import pandas as pd
import numpy as np
import os
from astropy.io import fits
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import RandomFlip, RandomRotation
from tensorflow.keras.models import Sequential
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from astropy.stats import sigma_clip

def sigma_clipping(X, sigma):
    mask = sigma_clip(X, maxiters=5, sigma=sigma, axis= (1,2) ).mask
    X[~ mask] = 0
    return X

def get_train_val_dataset(cf):
    return _get_dataset("train_val", cf)

def get_test_dataset(cf):
    return _get_dataset("test", cf)
    
def _get_dataset(kind, cf):
    if cf.SIGMA_CLIPPING:
        filekind = "clip"
    else:
        filekind = "no_clip"

    return [
        np.load(f"{cf.DATASET_DIR}X_{kind}_{filekind}.npy"),
        np.load(f"{cf.DATASET_DIR}y_{kind}.npy"),
        np.load(f"{cf.DATASET_DIR}names_{kind}.npy")
    ]


def get_data(cf):
    df = pd.read_csv(cf.DATA_CSV, index_col=0)
    df = df[df["Type"].isin(cf.TYPES)]

    if cf.MATCH_PIXELS:
        im_dir = "same_pixels"
    elif len(cf.SURVEYS) > 1:
        print("Different pixel sizes not supported for multiple surveys.")
        im_dir = "same_pixels"
    else:
        im_dir = "different_pixels"

    fits_folder = f"{cf.FITS_DIR}/{im_dir}/"
    relevant_sources = df.index.to_list()
    relevant_files = [x for x in os.listdir(fits_folder) if x.split("_")[1] in cf.SURVEYS]

    y, x = fits.getdata(fits_folder + relevant_files[0], memmap=False).shape
    data_images = np.zeros((len(relevant_sources), y, x, len(cf.SURVEYS)))
    names, labels = [],[]

    for i, source in enumerate(relevant_sources):
        names.append(source)
        labels.append(df.loc[source, "Type"])
        files = sorted([x for x in relevant_files if x.startswith(f"{source}_")])
        for j, f in enumerate(files):
            data_images[i, :, :, j] = fits.getdata(fits_folder + f, memmap=False)

    return data_images, np.array(labels), np.array(names)


def check_empty_image_error(X, y, sources):
    empty_indices = []
    len_channels = X.shape[-1]
    for i, image in enumerate(X):
        for channel in range(len_channels):
            if np.isnan(image[:,:,channel]).all():
                empty_indices.append(i)
                break
    
    if len(empty_indices) > 0:
        raise ValueError(f"Found {len(empty_indices)} empty images. These sources are : {sources[empty_indices]}")
    

def fill_partial_images(X, sources):
    cut_off_sources = set()
    for i in range(X.shape[0]):
        for channel in range(X.shape[-1]):
            im = X[i,:,:,channel]
            if np.isnan(im).any():
                min_val = np.nanmin(im)
                X[i,:,:,channel] = np.nan_to_num(im, nan=min_val)
                cut_off_sources.add(sources[i])

    if len(cut_off_sources) > 0:
        print("!!!!!!!!!!!!!!\tFound", len(cut_off_sources), "sources with NaN values")
        print("!!!!!!!!!!!!!!\tFilled in NaN values with lowest pixel value relative to the image")
        print("!!!!!!!!!!!!!!\tSources with NaN values:", cut_off_sources)


def split_data(X, y, names, split=0.2, seed=0):
    return train_test_split(X, y, names, test_size=split, random_state=seed, stratify=y)

def get_folds(X, y, val_split, random_state):
    stratified_kfold = StratifiedKFold(n_splits=int(1/val_split), random_state=random_state, shuffle=True)
    return [[t,v] for t,v in stratified_kfold.split(X, y)]


def generate_augmented_data(X, y, sources, cf):
    rotate_flip = Sequential([RandomFlip(),RandomRotation(1)]) # 1 being -360 to 360 degrees
    details = {}
    for morph in np.unique(y):
        num_of_sources = len(np.where(y == morph)[0])
        num_to_augment = cf.POST_AUGMENTATION_PER_TYPE - num_of_sources
        num_per_source = int(np.ceil(num_to_augment / num_of_sources))

        print(f"Augmenting {morph}: {num_of_sources}")
        print(f"\tNeed to augment at least {num_to_augment} images")
        print(f"\tAugmenting {num_per_source} images per source")
        print(f"\tThus, {num_per_source * num_of_sources} images will be added")

        details[morph] = num_per_source

    with open(cf.SAVE_DIR+"augmented_images.txt", "wb") as f:

        for i in range(len(y)):
            print(f"Augmenting... {i+1}/{len(y)}", end='\r')
            aug_images = []

            for n in range(details[y[i]]):
                image = np.array(rotate_flip(X[i]))
                aug_images.append(image)

            images = crop_normalize(np.array(aug_images), cf.CROP_SIZE, sources)

            if cf.SIGMA_CLIPPING:
                images = sigma_clipping(images, 3)

            np.save(f, images)
            
    np.save(cf.SAVE_DIR+"y_augmented.npy", y)


def generate_fourier_data(X, cf):
    return None


def crop_normalize(X, crop_size, names, only_crop=False):
    if len(X.shape) == 3:
        X = np.expand_dims(X, axis=0)

    width, height, channel = X[0].shape
    cropped = X[:, width//2 - crop_size//2:width//2 + crop_size//2, height//2 - crop_size//2:height//2 + crop_size//2, :]

    if only_crop:
        return cropped

    # Makes sure that there are no blank images before continuing.
    for i, image in enumerate(cropped):
        maxi = np.max(image)
        mini = np.min(image)
        if maxi == mini:
            raise ValueError("Image has no variation in pixel values after cropping it.", names[i])
        
    minis = np.min(cropped, axis=(1,2), keepdims=True)
    maxis = np.max(cropped, axis=(1,2), keepdims=True)

    for i in range(cropped.shape[0]):
        cropped[i] = (cropped[i] - minis[i]) / (maxis[i] - minis[i])
    return cropped


def build_dataset(X, y, names, indices, size_per_type, seed, save_dir, sigma, cf):

    print("Building Dataset...")

    morphological_detail = []
    additional_length = 0
    for morph in np.unique(y):
        morph_indices = indices[np.where(y[indices] == morph)[0]]

        aug_per_source = (size_per_type - len(morph_indices)) // len(morph_indices)
        additional_aug = (size_per_type - len(morph_indices)) % len(morph_indices)

        additional_length += (len(morph_indices) * aug_per_source + additional_aug)
        morphological_detail.append((morph, morph_indices, aug_per_source, additional_aug))
    
    # # -------------------------------------------XXX-------------------------------------------#
    # import matplotlib.pyplot as plt
    # for i in range(len(X)):
    #     plt.imshow(X[i,:,:,0])
    #     plt.colorbar()
    #     plt.title(f"{y[i]} - {names[i]}")
    #     plt.show()
    # # -------------------------------------------XXX-------------------------------------------#
    
    X = np.concatenate((X[indices], np.zeros((additional_length, X.shape[1], X.shape[2], X.shape[3]))), axis=0)
    y = np.concatenate((y[indices], np.empty(additional_length, dtype=str)), axis=0)
    names = np.concatenate((names[indices], np.empty(additional_length, dtype=str)), axis=0)
    offset = len(indices)

    if sigma:
        clip_type = "clip"
    else:
        clip_type = "no_clip"

    for morph, morph_indices, aug_per_source, additional_aug in morphological_detail:
        chosen_additional = np.random.choice(len(morph_indices), additional_aug, replace=False)
        
        file_pointer = -1
        previous_index = -1
        # with open(save_dir+"augmented_images_" + clip_type + ".txt", "rb") as f:
        with open(cf.DATASET_DIR + "augmented_images_" + clip_type + ".txt", "rb") as f:

            for i in sorted(np.concatenate([morph_indices, morph_indices[chosen_additional]])):
                if i != previous_index:
                    data = np.load(f)
                    amount = aug_per_source
                    file_pointer += 1
                else:
                    amount = 1
                while file_pointer < i:
                    data = np.load(f)
                    file_pointer += 1

                # # -------------------------------------------XXX-------------------------------------------#
                # import matplotlib.pyplot as plt
                # print("Next Image", i)
                # print(data.shape)
                # for i in range(data.shape[0]):
                #     plt.imshow(data[i,:,:,0])
                #     plt.colorbar()
                #     plt.show()
                # -------------------------------------------XXX-------------------------------------------#
                
                X[offset:offset+amount] = data[:amount]
                y[offset:offset+amount] = morph
                names[offset:offset+amount] = names[i]
                offset += amount
                previous_index = i

    print("Shuffling...")
    return shuffle(X, y, names, random_state=seed)   