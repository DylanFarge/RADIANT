from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, InputLayer
from tensorflow.keras import Sequential
from tensorflow.keras import initializers, regularizers


def ConvXpress(random_state,input_shape,num_classes,regularisation):
    """Author: https://github.com/ezrafielding/Galaxy10-convXpress/blob/main/define_models.py
    Defines the ConvXpress Model.

    Args:
        random_state: Seed for the Random function.
        input_shape: The expected input shape for the model.
        num_classes: The size of the output layer / nmumber of classes for dataset.

    Returns:
        ConvXpress Layers.
    """
    model = Sequential()
    model.add(InputLayer(shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3, 3),strides=2,activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3),strides=2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(regularisation), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    print("Returning ConvXpress")
    return model