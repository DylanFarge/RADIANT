import numpy as np
import process
import os
from prints import print_distribution, print_train_val_distribution
from tensorflow.keras.optimizers import Adam # type: ignore
from models import ConvXpress

def _validate_directory(cf):
    if os.path.exists(cf.SAVE_DIR) and not cf.IGNORE_WARNINGS:
        while True:
            response = input(f"Directory {cf.SAVE_DIR} already exists. Do you want to overwrite it? (y/n): ")
            if response.lower() == "y":
                break
            elif response.lower() == "n":
                exit()
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
    elif not os.path.exists(cf.SAVE_DIR):
        os.makedirs(cf.SAVE_DIR)


def _format_config(cf):
    surveys = cf.SURVEYS
    cf.SURVEYS = [surveys] if isinstance(surveys, str) else surveys
    if len(cf.SURVEYS) > 1:
        cf.BATCH_SIZE = 16
    else:
        cf.BATCH_SIZE = 32
    cf.SAVE_DIR = f"{cf.SAVE_DIR.rstrip('/')}/{'.'.join(cf.SURVEYS)}_{'same' if cf.MATCH_PIXELS else 'diff'}_{cf.POST_AUGMENTATION_PER_TYPE}a_{cf.EPOCHS}e_{cf.LEARNING_RATE}l_{cf.REGULARISATION}r/"


def run_experiment(X_train_val, y_train_val, X_test, y_test, names_train_val, folds, cf):
    losses = []
    for kfold, (train_idx, val_idx) in enumerate(folds):
        print(f">>>>>Training Fold {kfold+1}<<<<<<")

        if np.isnan(X_train_val).any():
            raise ValueError("NaN values found in training/validation data")

        X_train, y_train, names_train = process.build_dataset(X_train_val, y_train_val, names_train_val, train_idx, cf.POST_AUGMENTATION_PER_TYPE - int(cf.POST_AUGMENTATION_PER_TYPE * cf.VAL_SPLIT), cf.SEED, cf.SAVE_DIR, cf.SIGMA_CLIPPING, cf)
        X_val, y_val, names_val = process.build_dataset(X_train_val, y_train_val, names_train_val,  val_idx, int(cf.POST_AUGMENTATION_PER_TYPE * cf.VAL_SPLIT), cf.SEED, cf.SAVE_DIR, cf.SIGMA_CLIPPING, cf)

        print_train_val_distribution(X_train, y_train, X_val, y_val)

        print("Creating New Model...")
        model = ConvXpress(
            random_state=cf.SEED,
            input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]),
            num_classes=len(np.unique(y_train_val)),
            regularisation = cf.REGULARISATION
        )
        model.compile(
            optimizer=Adam(learning_rate=cf.LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
            # metrics=["accuracy", "f1_score"] #* This causes model.fit to throw an error. Need to reshape things. 
        )
        
        encoded_y_train = np.array([cf.LOOKUP[x] for x in y_train])
        encoded_y_val = np.array([cf.LOOKUP[x] for x in y_val])
        print("BATCH SIZE:", cf.BATCH_SIZE)
        loss = model.fit(
                    X_train, encoded_y_train,
                    validation_data=(X_val, encoded_y_val),
                    epochs=cf.EPOCHS,
                    batch_size=cf.BATCH_SIZE,
                ).history

        losses.append(loss)

        encoded_y_pred_test = model.predict(X_test)
        encoded_y_pred_train = model.predict(X_train)
        encoded_y_pred_val = model.predict(X_val)

        np.save(f"{cf.SAVE_DIR}y_pred_test_{kfold+1}", encoded_y_pred_test)
        np.save(f"{cf.SAVE_DIR}y_pred_train_{kfold+1}", encoded_y_pred_train)
        np.save(f"{cf.SAVE_DIR}y_pred_val_{kfold+1}", encoded_y_pred_val)


        np.save(f"{cf.SAVE_DIR}loss_fold_{kfold+1}", loss)
        # np.save(f"{cf.SAVE_DIR}X_train_fold_{kfold+1}", X_train)
        np.save(f"{cf.SAVE_DIR}y_train_fold_{kfold+1}", y_train)
        # np.save(f"{cf.SAVE_DIR}X_val_fold_{kfold+1}", X_val)
        np.save(f"{cf.SAVE_DIR}y_val_fold_{kfold+1}", y_val)
    
    np.save(f"{cf.SAVE_DIR}lookup", cf.LOOKUP)
    


def main(**kwargs):
    class Config:
        def __init__(self, **kwargs):
            # make keys properties
            self.__dict__.update(kwargs)

    cf = Config(**kwargs)

    np.random.seed(cf.SEED)

    _format_config(cf)
    _validate_directory(cf)

    X_train_val, y_train_val, names_train_val = process.get_train_val_dataset(cf=cf)
    X_test, y_test, names_test = process.get_test_dataset(cf=cf)    

    folds = process.get_folds(X_train_val, y_train_val, cf.VAL_SPLIT, cf.SEED)

    for i, (train_idx, val_idx) in enumerate(folds):
        print_train_val_distribution(X_train_val[train_idx], y_train_val[train_idx], X_train_val[val_idx], y_train_val[val_idx], name=f"Fold {i+1}")

    cf.LOOKUP = {morph: i for i, morph in enumerate(np.unique(y_train_val))}
    run_experiment(X_train_val, y_train_val, X_test, y_test, names_train_val, folds, cf)

if __name__ == "__main__": 
    
    experiments = [
        # learning rate, regularisation

    #     ( 1.00e-06, 1.01e+01 ),   
    #     ( 1.00e-06, 5.10e+00 ),   
    #     ( 1.00e-06, 1.00e-01 ),   
    #     ( 2.00e-03, 1.01e+01 ),   
    #     ( 2.00e-03, 1.00e-01 ),   
    #     ( 1.75e-03, 3.00e+00 ),   
    #     ( 1.75e-03, 1.00e+00 ),   
    #     ( 1.75e-03, 8.00e+00 ),   
    #     ( 1.75e-03, 6.00e+00 ),   
    #     ( 1.50e-03, 5.00e+00 ),    
    #     ( 1.25e-03, 6.00e+00 ),    
    #     ( 1.25e-03, 1.00e+00 ),    
    #     ( 1.00e-03, 1.01e+01 ),    
    #     ( 1.50e-03, 9.00e+00 ),    
    #     ( 1.50e-03, 7.00e+00 ),    
    #     ( 1.00e-03, 5.10e+00 ),    
    #     ( 1.25e-03, 8.00e+00 ),    
    #     ( 1.50e-03, 2.00e+00 ),    
    #     ( 1.25e-03, 3.00e+00 ),    
    #     ( 1.50e-03, 4.00e+00 ),    
        ( 2.50e-04, 8.00e+00 ),    
    #     ( 2.50e-04, 6.00e+00 ),    
    #     ( 1.00e-03, 1.00e-01 ),    
    #     ( 7.50e-04, 8.00e+00 ),    
    #     ( 7.50e-04, 6.00e+00 ),    
    #     ( 7.50e-04, 3.00e+00 ),    
    #     ( 7.50e-04, 1.00e+00 ),    
    #     ( 5.00e-04, 7.00e+00 ),    
    #     ( 5.00e-04, 4.00e+00 ),    
    #     ( 5.00e-04, 9.00e+00 ),    
    #     ( 2.50e-04, 1.00e+00 ),    
    #     ( 2.50e-04, 3.00e+00 ),    
    #     ( 5.00e-04, 5.00e+00 ),    
    #     ( 5.00e-04, 2.00e+00 ),
    ]

    # experiments = []
    # for lr in [1.000e-06, 2.510e-04, 5.010e-04, 7.510e-04, 1.001e-03, 1.251e-03, 1.501e-03, 1.751e-03]:
    #     for reg in [1.0000e-03, 2.0010e-00, 4.0010e-00, 6.0010e-00, 8.0010e-00, 1.0001e+01]:
    #         experiments.append((lr, reg))

    SAVE_DIR = "MachineLearning/results/"
    DATASET_DIR = "AugmentData/results/"
    SURVEY = ["FIRST"]
    # SURVEY = ["LOFAR"]
    # SURVEY = ["NVSS"]
    # SURVEY = ["FIRST", "LOFAR", "NVSS"]
    MATCH_PIXELS = True
    POST_AUGMENTATION_PER_TYPE = 3000
    EPOCHS = 30
    SIGMA_CLIPPING = True

    # ---------------------------Check used in conjunction with manager.py------------------------------------------
    # 
    counter = 0
    prev_lr = 0
    prev_reg = 0
    for i, (lr, reg) in enumerate(experiments):    
        save_dir = f"{SAVE_DIR.rstrip('/')}/{'.'.join(SURVEY)}_{'same' if MATCH_PIXELS else 'diff'}_{POST_AUGMENTATION_PER_TYPE}a_{EPOCHS}e_{lr}l_{reg}r/"
        if os.path.exists(save_dir):
            prev_lr = lr
            prev_reg = reg
            print("\nPath exists:", save_dir)
            counter += 1
        elif counter > 0:
            print("\nPath does not exist:", save_dir)
            save_dir = f"{SAVE_DIR.rstrip('/')}/{'.'.join(SURVEY)}_{'same' if MATCH_PIXELS else 'diff'}_{POST_AUGMENTATION_PER_TYPE}a_{EPOCHS}e_{prev_lr}l_{prev_reg}r/"
            os.system(f"rm -r {save_dir}")
            print("Deleted:", save_dir)
            counter -= 1
            break
        else:
            break
    experiments = experiments[counter:]
    # --------------------------------------------------------------------------------------------------------------

    for i, (lr, reg) in enumerate(experiments):

        print(f"Running experiment {i+1}/{len(experiments)} with learning rate {lr} and regularisation {reg}")
        main(
            SEED = 100,
            TEST_SPLIT = 0.2,
            VAL_SPLIT = 0.2,
            CROP_SCALE = 128/300,
            TYPES=["COMPACT", "FRI", "FRII"],
            IGNORE_WARNINGS = False,
            DATA_CSV="ConstructData/RADCAT_ML.csv",
            FITS_DIR="downloaded_images",
            
            POST_AUGMENTATION_PER_TYPE= POST_AUGMENTATION_PER_TYPE,    
            EPOCHS = EPOCHS,
            LEARNING_RATE = lr,
            REGULARISATION = reg,
            SAVE_DIR = SAVE_DIR,
            SURVEYS = SURVEY,
            MATCH_PIXELS=MATCH_PIXELS,
            SIGMA_CLIPPING =  SIGMA_CLIPPING,
            DATASET_DIR = DATASET_DIR,
        )