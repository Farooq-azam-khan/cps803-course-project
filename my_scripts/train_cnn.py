import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from helpers import (
                    load_train_val_data, 
                    plot_accuracy, 
                    plot_loss, 
                    configure_gpu_memory_growth, 
                    make_experiment_dir, 
                    save_history, 
                    get_augmentation_layer
                    ) 

IMG_PIXELS = 224
image_size = (IMG_PIXELS, IMG_PIXELS)
batch_size = 16
model_type = 'CNN_L2'
epochs = 200
learning_rate = 1e-3

def get_train_val_ds():
    
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory='../data/train',
    validation_split=0.2,
    subset="training",
    label_mode="categorical",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    shuffle=True)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory='../data/train',
        validation_split=0.2,
        subset="validation",
        label_mode="categorical",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True)
    
    return (train_ds, val_ds)

def build_cnn_model(num_classes, num_dense_layers):
    inputs = keras.Input(shape=(IMG_PIXELS, IMG_PIXELS, 3))
    
    model = keras.Sequential()
    model.add(inputs)
    model.add(layers.Rescaling(1./255))
    model.add(layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"))
    model.add(layers.experimental.preprocessing.RandomRotation(0.2))

    model.add(layers.Conv2D(32, 3, padding="same", activation="relu", input_shape=(IMG_PIXELS, IMG_PIXELS, 3)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, 3, padding="same", activation="relu"))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128, 3, padding="same", activation="relu"))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(512, 3, padding="same", activation="relu"))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    
    for _ in range(num_dense_layers):
        model.add(layers.Dense(units=64, activation="relu", kernel_regularizer='l2'))
        model.add(layers.Dropout(0.5))
        
        
    model.add(layers.Dense(units=num_classes, activation="softmax", kernel_regularizer='l2'))
    
    model.build()
    model.compile(
        optimizer='adam',
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def main():
    configure_gpu_memory_growth()
    print('Loading Data...')
    train_ds, val_ds = get_train_val_ds()
    
    model = build_cnn_model(num_classes=6, num_dense_layers=1)
    
    time_stamp = int(time.time())
    experiment_dir = make_experiment_dir(model_type, str(time_stamp)) # have access to 'experiment_dir/models', 'experiment_dir/plots'
    experiment_dir_models = experiment_dir / 'models'
    experiment_dir_plots = experiment_dir / 'plots'
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=experiment_dir_models / 'L2_{epoch}_{val_accuracy:.2f}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]
    
    
    hist = model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks, verbose=1)
    save_history(hist.history, experiment_dir_plots)
    plot_accuracy(hist, experiment_dir_plots, batch_size, image_size, model_type, epochs, save_as_tex=True)
    plot_loss(hist, experiment_dir_plots, batch_size, image_size, model_type, epochs, save_as_tex=True)

    print('Evaluating Model...')
    evaluate_model_on_test_data(model, image_size, model_type, experiment_dir_plots)
   
if __name__ == '__main__':
    main()

