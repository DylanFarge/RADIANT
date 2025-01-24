import os
import pandas as pd
from astropy.io import fits
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import RandomFlip, RandomRotation #type: ignore
from tensorflow.keras.models import Sequential #type: ignore
from astropy.stats import sigma_clip

def print_distribution(dataset, dataset_name, total_images=None, spacing=20):
    total_images = len(dataset) if total_images is None else total_images
    print(f"\n--- {dataset_name} Distribution ---")
    unique, counts = np.unique(dataset, return_counts=True)
    print(f"{'Morphology':^{spacing}}|{'Count':^{spacing}}|{'Percentage':^{spacing}}")
    print(f"{'-'*spacing}|{'-'*spacing}|{'-'*spacing}")
    for i in range(len(unique)):
        print(f"{unique[i]:^{spacing}}|{counts[i]:^{spacing}}|{f'{counts[i]/len(dataset)*100:.2f}%':^{spacing}}")
    print(f"{'-'*spacing}|{'-'*spacing}|{'-'*spacing}")
    print(f"Total Images: {len(dataset)}/{total_images}\t{len(dataset)/total_images*100:.2f}%\n")


def check_empty_image_error(X, sources):
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
        print(f"Filled {len(cut_off_sources)} partial images. These sources are : {cut_off_sources}")
        np.save("ConstructData/results/filled_partial_names.npy", list(cut_off_sources))
        exit()

def get_RADCAT_data(catalog_file, fits_folder):
    df = pd.read_csv(catalog_file, index_col=0)
    df = df[df["Type"].isin(["COMPACT", "FRI", "FRII"])]
    
    relevant_sources = df.index.to_list()
    relevant_files = os.listdir(fits_folder)
    
    data_images = np.zeros((len(relevant_sources), 300, 300, 3))
    names, labels = [],[]
    
    for i, source in enumerate(relevant_sources):
        
        names.append(source)
        labels.append(df.loc[source, "Type"])
        files = sorted([x for x in relevant_files if x.startswith(f"{source}_")])
        
        for j, file in enumerate(files):
            data_images[i, :, :, j] = fits.getdata(fits_folder + file, memmap=False)

    return data_images, np.array(labels), np.array(names)


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


def generate_augmented_data(X, y, names, save_dir):
    rotate_flip = Sequential([RandomFlip(),RandomRotation(1)]) # 1 being -360 to 360 degrees
    details = {}
    for morph in np.unique(y):
        num_of_sources = len(np.where(y == morph)[0])
        num_to_augment = 3000 - num_of_sources
        num_per_source = int(np.ceil(num_to_augment / num_of_sources))

        print(f"Augmenting {morph}: {num_of_sources}")
        print(f"\tNeed to augment at least {num_to_augment} images")
        print(f"\tAugmenting {num_per_source} images per source")
        print(f"\tThus, {num_per_source * num_of_sources} images will be added")

        details[morph] = num_per_source

    with open(save_dir + "augmented_images_no_clip.txt", "wb") as f_no_clip:
        with open(save_dir + "augmented_images_clip.txt", "wb") as f_clip:

            for i in range(len(names)):
                print(f"Augmenting... {i+1}/{len(names)}", end='\r')
                aug_images = []

                for n in range(details[y[i]]):
                    image = np.array(rotate_flip(X[i]))
                    aug_images.append(image)
                
                images = crop_normalize(np.array(aug_images), 128, [names[i]])
                np.save(f_no_clip, images)

                images = sigma_clipping(images, 3)
                np.save(f_clip, images)

            
    np.save(save_dir + "y_augmented.npy", y)
    np.save(save_dir + "names_augmented.npy", names)
            
        
def sigma_clipping(X, sigma):
    mask = sigma_clip(X, maxiters=5, sigma=sigma, axis= (1,2) ).mask
    X[~ mask] = 0
    return X


if __name__ == "__main__":

    save_dir = "ConstructData/results/"

    # X, y, names = get_RADCAT_data(catalog_file="ConstructData/results/RADCAT_ML.csv",fits_folder="/mnt/c/Users/dylan/Documents/downloaded_images/same_pixels/")
    X, y, names = get_RADCAT_data(catalog_file="ConstructCatalog/results/RADCAT_ML.csv",fits_folder="downloaded_images/same_pixels/")
    
    check_empty_image_error(X, names)
    fill_partial_images(X, names)
    X_train_val, X_test, y_train_val, y_test, names_train_val, names_test = train_test_split(X, y, names, test_size=0.2, random_state=100)

    for key, val in {"TrainVal": y_train_val, "Testing": y_test}.items():
        print_distribution(val, key, total_images=len(y))
        
    del X, y, names
    
    generate_augmented_data(X_train_val, y_train_val, names_train_val, save_dir)

    X_train_val = crop_normalize(X_train_val, 128, names_train_val)
    X_test = crop_normalize(X_test, 128 , names_test)

    np.save(save_dir + "X_train_val_no_clip.npy", X_train_val)
    np.save(save_dir + "X_test_no_clip.npy", X_test)

    X_train_val = sigma_clipping(X_train_val, 3)
    X_test = sigma_clipping(X_test, 3)

    np.save(save_dir + "X_train_val_clip.npy", X_train_val)
    np.save(save_dir + "X_test_clip.npy", X_test)

    np.save(save_dir + "y_train_val.npy", y_train_val)
    np.save(save_dir + "y_test.npy", y_test)
    np.save(save_dir + "names_train_val.npy", names_train_val)
    np.save(save_dir + "names_test.npy", names_test)
    
    print("Done")    