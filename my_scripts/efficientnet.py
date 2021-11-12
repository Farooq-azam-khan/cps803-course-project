#https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers 
import tensorflow as tf
import time 
from tensorflow import keras

from helpers import (
                    load_train_val_data, 
                    plot_accuracy, 
                    plot_loss
                    )   

IMG_PIXELS = 224
image_size = (IMG_PIXELS, IMG_PIXELS)
batch_size = 16
model_type = 'EfficientNet'
epochs = 15

img_augmentation = tf.keras.models.Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1),
    ],
    name='img_augmentation',
)

def build_efficient_net_model(num_classes):
    inputs = layers.Input(shape=(IMG_PIXELS,IMG_PIXELS, 3))
    x = img_augmentation(inputs)
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights='imagenet')

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name='avg_pool')(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name='top_dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='pred')(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name='EfficientNet')
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy']
    )
    return model

def main():
    print('Loading Data...')
    train_ds, val_ds = load_train_val_data(image_size=image_size, batch_size=batch_size)

    model = build_efficient_net_model(num_classes=6)
    time_stamp = int(time.time())

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath='./models/EfficientNet/simple_{epoch}_{val_accuracy:.2f}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'# max becuase we want to save based on val_accuracy (if loss then min)
        )
    ]
    
    hist = model.fit(train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds, verbose=2)
    plot_accuracy(hist, time_stamp, batch_size, image_size, model_type, epochs)
    plot_loss(hist, time_stamp, batch_size, image_size, model_type, epochs)

if __name__ == '__main__':
    main()