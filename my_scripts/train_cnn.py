import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import pathlib
import time 
from keras.utils.vis_utils import plot_model

import matplotlib.pyplot as plt

# Hyper parameters
image_size = (200,200) #(384, 512)
batch_size = 16
epochs = 10
learning_rate = 1e-3
model_type = 'cnn'
base_dir = pathlib.Path('..')
data_train = base_dir / 'data' / 'train'
target_classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
NUM_OF_DENSE_LAYS = 2



def make_cnn_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    model = keras.Sequential()
    model.add(inputs)
    model.add(layers.Rescaling(1./255))

    model.add(layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=input_shape))
    # 1 dense layer of 512 size
    model.add(layers.Flatten())
    # 20 Layers 
    for _ in range(NUM_OF_DENSE_LAYS):
        model.add(layers.Dense(units=512, activation="relu"))

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
        subset="training",
        label_mode="categorical",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_train,
        validation_split=0.2,
        subset="validation",
        label_mode="categorical",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )
    return train_ds, val_ds


def plot_loss(history, time_stamp):
    plt.figure()
    plt.title(f'Simple NN Loss - batch size {batch_size}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(range(1, epochs+1), history.history['val_loss'])
    plt.plot(range(1, epochs+1), history.history['loss'])
    plt.legend(['Validation Loss', 'Training Loss'])
    
    plt.savefig(f'./plots/{model_type}/{time_stamp}_loss_size_{image_size[0]}_{image_size[1]}.jpeg')

def plot_accuracy(history, time_stamp):
    plt.figure()
    plt.title(f'Simple NN Accuracy - batch size {batch_size}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(range(1, epochs+1), history.history['val_accuracy'])
    plt.plot(range(1, epochs+1), history.history['accuracy'])
    plt.legend(['Validation Accuracy', 'Training Accuracy'])
    plt.savefig(f'./plots/{model_type}/{time_stamp}_acc_size_{image_size[0]}_{image_size[1]}.jpeg')
    
def main():
    configure_gpu_memory_growth()
    train_ds, val_ds = load_train_val_data()

    print('Making Model')
    model = make_cnn_model(input_shape=image_size + (3,), num_classes=len(target_classes))
    model.build()
    print(model.summary())
    
    print('Plotting Model')
    curr_time = int(time.time())
    unique_plot_name = f'./plots/{model_type}/{curr_time}_simple.png'
    # docs: https://www.tensorflow.org/api_docs/python/tf/keras/utils/plot_model
    #plot_model(model, to_file=unique_plot_name, show_shapes=True)

    print('Training Model')
    time_start = time.time()
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
    time_end = time.time()

    print('Time Elapsed: ', time_end - time_start)

    print('Creating Plots')
    plot_loss(history, curr_time)
    plot_accuracy(history,curr_time)
    


if __name__ == '__main__':
    main()

