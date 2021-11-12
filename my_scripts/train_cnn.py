import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pathlib
import time 
from keras.utils.vis_utils import plot_model

import matplotlib.pyplot as plt

import helpers 

image_size = (100, 100) #(384, 512)
batch_size = 8
epochs = 5
learning_rate = 1e-3
model_type = 'cnn'
base_dir = pathlib.Path('..')
data_train = base_dir / 'data' / 'train'
target_classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


def make_cnn_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    model = keras.Sequential()
    model.add(inputs)
    model.add(layers.Rescaling(1./255))
    # 1 Conv2d layer  and one dense layer of 512 size
    model.add(layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=input_shape))
    # model.add(layers.Conv2D(64, 3, padding="same", activation="relu"))
    # model.add(layers.Conv2D(64, 3, padding="same", activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(num_classes, activation="softmax"))
    return model 

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
    
def load_train_val_data():

    print('Loading data')
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        '../data/train',
        validation_split=0.2,
        subset='training',
        label_mode='categorical',
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_train,
        validation_split=0.2,
        subset='validation',
        label_mode='categorical',
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )
    return train_ds, val_ds

def plot_loss(history, time_stamp):
    plt.figure()
    plt.title('Simple NN Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(range(1, epochs+1), history.history['val_loss'])
    plt.plot(range(1, epochs+1), history.history['loss'])
    plt.legend(['Validation Loss', 'Training Loss'])
    file_name = helpers.get_plot_file_name(model_type, 'loss', time_stamp, image_size)
    plt.savefig(file_name)

def plot_accuracy(history, time_stamp):
    plt.figure()
    plt.title('Simple NN Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(range(1, epochs+1), history.history['val_accuracy'])
    plt.plot(range(1, epochs+1), history.history['accuracy'])
    plt.legend(['Validation Accuracy', 'Training Accuracy'])
    file_name = helpers.get_plot_file_name(model_type, 'acc', time_stamp, image_size)
    plt.savefig(file_name)
    
def main():
    configure_gpu_memory_growth()
    train_ds, val_ds = load_train_val_data()

    print('Making Model')
    model = make_cnn_model(input_shape=image_size + (3,), num_classes=len(target_classes))
    model.build()
    print(model.summary())
    
    print('Plotting Model')
    curr_time = int(time.time())
    unique_plot_name = f'./plots/simple_{model_type}_{curr_time}.png'
    # docs: https://www.tensorflow.org/api_docs/python/tf/keras/utils/plot_model
    plot_model(model, to_file=unique_plot_name, show_shapes=True)

    print('Training Model')
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath='./models/cnn/simple_{epoch}_{val_accuracy:.2f}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'# max becuase we want to save based on val_accuracy (if loss then min)
        )
    ]

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    history = model.fit(
        train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
    )

    print('Creating Plots')
    plot_loss(history, curr_time)
    plot_accuracy(history,curr_time)
    


if __name__ == '__main__':
    main()
