
from tensorflow.keras import layers 
import tensorflow as tf
import time 
from tensorflow import keras
from tensorflow.keras.applications import ResNet50#https://keras.io/api/applications/resnet/
from helpers import (
                    load_train_val_data, 
                    plot_accuracy, 
                    plot_loss
                    )   

IMG_PIXELS = 224
image_size = (IMG_PIXELS, IMG_PIXELS)
batch_size = 16
model_type = 'resnet'
epochs = 10#50#100
learning_rate = 1e-5

img_augmentation = tf.keras.models.Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1),
    ],
    name='img_augmentation',
)


def configure_gpu_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print('Found GPU on Device, configuring memory growth')
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def build_resnet_net_model(num_classes):
    inputs = layers.Input(shape=(IMG_PIXELS,IMG_PIXELS, 3))
    x = img_augmentation(inputs)
    #model = EfficientNetB0(include_top=False, input_tensor=inputs, weights='imagenet')
    model = ResNet50(include_top=False, input_tensor=inputs, weights='imagenet')
    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name='avg_pool')(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    #x = layers.Dropout(top_dropout_rate, name='top_dropout')(x)
    # taper of the layer nodes
    x = layers.Dense(800, name='dense_800', activation='relu')(x)
    x = layers.Dense(600, name='dense_600', activation='relu')(x)
    x = layers.Dense(100, name='dense_100', activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='pred')(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name='EfficientNet')
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy']
    )
    return model

def main():
    configure_gpu_memory_growth()
    print('Loading Data...')
    train_ds, val_ds = load_train_val_data(image_size=image_size, batch_size=batch_size)

    model = build_resnet_net_model(num_classes=6)
    time_stamp = int(time.time())

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath='./models/resnet/en_dense_no_dropout_{epoch}_{val_accuracy:.2f}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'# max becuase we want to save based on val_accuracy (if loss then min)
        )
    ]
    
    hist = model.fit(train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds, verbose=2)
    import numpy as np 
    np.save('./models/resnet/hist_'+str(time_stamp)+'.npy', hist.history)
    plot_accuracy(hist, time_stamp, batch_size, image_size, model_type, epochs)
    plot_loss(hist, time_stamp, batch_size, image_size, model_type, epochs)


if __name__ == '__main__':
    main()