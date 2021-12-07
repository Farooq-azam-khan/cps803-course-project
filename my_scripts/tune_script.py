import keras_tuner as kt

import tensorflow as tf 
import numpy as np 

from tensorflow.keras import layers 
from tensorflow.keras import regularizers
from tensorflow.keras import metrics 

# Prebuild model Imports 
from tensorflow.keras.applications import (EfficientNetB0, 
                                           EfficientNetB1,
                                           EfficientNetB2,
                                           EfficientNetB3,
                                           EfficientNetB4,
                                           )
from tensorflow.keras.applications import (ResNet101, ResNet50)
# custom imports 
from helpers import get_augmentation_layer

# Hyper Parameters
model_type = {
    'EfficientNet-B0': {
        'image_size': 224,
        'model_class': EfficientNetB0,
    }, 
    'EfficientNet-B1': {
        'image_size': 240,
        'model_class': EfficientNetB1,
    }, 
    'EfficientNet-B2': {
        'image_size': 260,
        'model_class': EfficientNetB2,
    }, 
    'EfficientNet-B3': {
        'image_size': 300,
        'model_class': EfficientNetB3,
    }, 
    'EfficientNet-B4': {
        'image_size': 380,
        'model_class': EfficientNetB4,
    }, 
    'Resnet50': {
        'image_size': 224,
        'model_class': ResNet50,
    },  
    'Resnet101': {
        'image_size': 224,
        'model_class': ResNet101,
    },
}
model_type_choice = [key for key in model_type.keys()]

batch_size = [16, 32, 64, 128]
learning_rate = [1e-3, 1e-4, 1e-5]
regularization_rate = [0, 1e-3, 1e-4, 1e-5] # 0 means dont regularize
# num_dense_layers = [1, 2, 3, 4]
num_classes = 5 

def model_bulider(hp):
    # pick a model type 
    hp_model_dict = model_type[hp.Choice('model_type', model_type_choice)]
    IMG_PIXELS = hp_model_dict['image_size']
    inputs = layers.Input(shape=(IMG_PIXELS,IMG_PIXELS, 3))
    x = get_augmentation_layer()(inputs)
    model = hp_model_dict['model_class'](include_top=False, input_tensor=inputs, weights='imagenet')
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

