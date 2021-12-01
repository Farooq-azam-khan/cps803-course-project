#https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers 
import tensorflow as tf
import time 
from tensorflow import keras

from helpers import (
                    load_train_val_data, 
                    plot_accuracy, 
                    plot_loss, 
                    configure_gpu_memory_growth, 
                    make_experiment_dir, 
                    save_history
                    )   

from evaluate_model import evaluate_model_on_test_data

IMG_PIXELS = 224
image_size = (IMG_PIXELS, IMG_PIXELS)
batch_size = 16
model_type = 'EfficientNetB0'
epochs = 5#50#100
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
    # new style of saving data: 
    # experiments/{model_type}/{timestamp}/models
    # experiments/{model_type}/{timestamp}/plots
    model = build_efficient_net_model(num_classes=6)
    time_stamp = int(time.time())
    experiment_dir = make_experiment_dir(model_type, str(time_stamp)) # have access to 'experiment_dir/models', 'experiment_dir/plots'
    experiment_dir_models = experiment_dir / 'models'
    experiment_dir_plots = experiment_dir / 'plots'
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=experiment_dir_models / 'en_dense_no_dropout_{epoch}_{val_accuracy:.2f}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'# max becuase we want to save based on val_accuracy (if loss then min)
        )
    ]
    
    hist = model.fit(train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds, verbose=1)
    save_history(hist.history, experiment_dir_plots)
    plot_accuracy(hist, experiment_dir_plots, batch_size, image_size, model_type, epochs, save_as_tex=True)
    plot_loss(hist, experiment_dir_plots, batch_size, image_size, model_type, epochs, save_as_tex=True)

    print('Evaluating Model...')
    evaluate_model_on_test_data(model, model_type, experiment_dir_plots)

if __name__ == '__main__':
    main()