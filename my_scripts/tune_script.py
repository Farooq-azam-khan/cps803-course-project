
import time 
import os 

#https://www.tensorflow.org/tutorials/keras/keras_tuner
import keras_tuner as kt

import tensorflow as tf 
import numpy as np 

from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras import regularizers


# Prebuild model Imports 
from tensorflow.keras.applications import (EfficientNetB0, 
                                           EfficientNetB1,
                                           EfficientNetB2,
                                           EfficientNetB3,
                                           EfficientNetB4,
                                           )
from tensorflow.keras.applications import (ResNet101, ResNet50)
# custom imports 
from helpers import (get_augmentation_layer, 
                    configure_gpu_memory_growth, 
                    load_train_val_data, 
                    save_history,
                    plot_accuracy,
                    plot_loss,
                    make_experiment_dir)

from evaluate_model import evaluate_model_on_test_data

# Hyper Parameters
model_type = {
    'EfficientNet-B0': {
        'image_pixels': 224,
        'model_class': EfficientNetB0,
    }, 
    # 'EfficientNet-B1': {
    #     'image_pixels': 240,
    #     'model_class': EfficientNetB1,
    # }, 
    # 'EfficientNet-B2': {
    #     'image_pixels': 260,
    #     'model_class': EfficientNetB2,
    # }, 
    # 'EfficientNet-B3': {
    #     'image_pixels': 300,
    #     'model_class': EfficientNetB3,
    # }, 
    # 'EfficientNet-B4': {
    #     'image_pixels': 380,
    #     'model_class': EfficientNetB4,
    # }, 
    'Resnet50': {
        'image_pixels': 224,
        'model_class': ResNet50,
    },  
    # 'Resnet101': {
    #     'image_pixels': 224,
    #     'model_class': ResNet101,
    # },
}
model_type_choice = [key for key in model_type.keys()]

batch_size = 16#[16, 32, 64, 128]
learning_rate = [1e-3, 1e-4, 1e-5]
regularization_rate = [0.00, 1e-3, 1e-4, 1e-5] # 0 means dont regularize
 
TUNER_EPOCHS = 5
BEST_TUNER_EPOCHS = 40


def model_builder_wrapper(model_class, image_pixels, num_classes):
    def model_bulider(hp):
        # pick a model type 
        inputs = layers.Input(shape=(image_pixels,image_pixels, 3))
        x = get_augmentation_layer()(inputs)
        model = model_class(include_top=False, input_tensor=inputs, weights='imagenet')
        model.trainable = False 

        x = layers.GlobalAveragePooling2D(name='avg_pool')(model.output)
        x = layers.BatchNormalization()(x)

        # pick regularization or not 
        hp_reqularization_rate = hp.Choice('regularization_rate', regularization_rate)
        
        # pick num of dense layers
        hp_dense_count = hp.Int('units', min_value=4, max_value=10, step=1)
        units_count = np.linspace(10, 900, hp_dense_count)[::-1]
        for uc in units_count:
            x = layers.Dense(int(uc), name=f'dense_{uc}', activation='relu', 
                kernel_regularizer=regularizers.l2(hp_reqularization_rate))(x)
        
        # final output layer
        outputs = layers.Dense(num_classes, activation='softmax', name='pred')(x)
        
        # Compile
        hp_learning_rate = hp.Choice('learning_rate', values=learning_rate)
        model = tf.keras.Model(inputs, outputs, name='EfficientNet')
        optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
        model.compile(
            optimizer=optimizer, 
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )
        return model
    return model_bulider

def main():
    configure_gpu_memory_growth()
    # for each model type, train the best hyper params
    for model_name in model_type:
        print('='*10, model_name, '='*10)
        model_config = model_type[model_name]
        # load the dataset 
        IMAGE_PIXELS = model_config['image_pixels']
        image_size = (IMAGE_PIXELS,IMAGE_PIXELS)
        print(f'Loading Data of {image_size} image size')
        train_ds, val_ds = load_train_val_data(image_size=image_size, batch_size=batch_size)
        num_classes = len(train_ds.class_names)
        kt_model_biulder_fn = model_builder_wrapper(model_config['model_class'], IMAGE_PIXELS, num_classes)
        # build the tuner
        tuner = kt.Hyperband(kt_model_biulder_fn,
                         objective='val_accuracy',
                         max_epochs=10,
                         factor=3,
                         directory=f'{model_name}_keras_tuner_garbage_class',
                         project_name=f'{model_name}_keras_tuner_garbage_class')

       

        callbacks = [ 
            # needed cause doing a search over the hyper params
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        ]

        tuner.search(train_ds, epochs=TUNER_EPOCHS, validation_data=val_ds, callbacks=callbacks, verbose=1)
        # Get the optimal hyperparameters
        best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f'Best hyperparameters: {best_hps.get_config()}')

        print('-'*10, 'Training the best model', '-'*10)
        model = tuner.hypermodel.build(best_hps)
        train_mest_model(model_name, model, train_ds, val_ds, image_size)

        

def train_mest_model(model_name, model, train_ds, val_ds, image_size):
    time_stamp = int(time.time())
    experiment_dir = make_experiment_dir(model_name, str(time_stamp))
    experiment_dir_models = experiment_dir / 'models'
    experiment_dir_plots = experiment_dir / 'plots'

    callbacks = [ 
                keras.callbacks.ModelCheckpoint(
                    filepath=experiment_dir_models / 'enb1_dense_no_dropout_{epoch}_{val_accuracy:.2f}.h5',
                    monitor='val_accuracy',
                    # save_best_only=True,
                    mode='max'# max becuase we want to save based on val_accuracy (if loss then min)
                ), 
                # needed cause doing a search over the hyper params
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
            ]

    
    hist = model.fit(train_ds, epochs=BEST_TUNER_EPOCHS, validation_data=val_ds, callbacks=callbacks)        

    
    
    save_history(hist.history, experiment_dir_plots)
    plot_accuracy(hist, experiment_dir_plots, batch_size, image_size, model_name, BEST_TUNER_EPOCHS, save_as_tex=True)
    plot_loss(hist, experiment_dir_plots, batch_size, image_size, model_name, BEST_TUNER_EPOCHS, save_as_tex=True)

    print('Evaluating Model...')
    evaluate_model_on_test_data(model, image_size, model_type, experiment_dir_plots)


if __name__ == '__main__':
    main()