import tensorflow as tf 
import pathlib 
import matplotlib.pyplot as plt

def get_plot_file_name(model_type, plot_type, time_stamp, image_size):
    """
    plot_type: 'acc', 'loss'
    model_type: 'cnn', 'nn', etc.
    image_size: (width, height)
    Returns the name of the plot file.
    """
    return f'./plots/{model_type}/{time_stamp}_{plot_type}_size_{image_size[0]}_{image_size[1]}.jpeg'


def load_train_val_data(image_size=(384, 512), batch_size=16):
    base_dir = pathlib.Path('..')
    data_train = base_dir / 'data' / 'train'
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


def plot_loss(history, time_stamp, batch_size, image_size, model_type, epochs):
    plt.figure()
    plt.title(f'Simple NN Loss - batch size {batch_size}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(range(1, epochs+1), history.history['val_loss'])
    plt.plot(range(1, epochs+1), history.history['loss'])
    plt.legend(['Validation Loss', 'Training Loss'])
    
    plt.savefig(f'./plots/{model_type}/{time_stamp}_loss_size_{image_size[0]}_{image_size[1]}.jpeg')


def plot_accuracy(history, time_stamp, batch_size, image_size, model_type, epochs):
    plt.figure()
    plt.title(f'Simple NN Accuracy - batch size {batch_size}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(range(1, epochs+1), history.history['val_accuracy'])
    plt.plot(range(1, epochs+1), history.history['accuracy'])
    plt.legend(['Validation Accuracy', 'Training Accuracy'])
    plt.savefig(f'./plots/{model_type}/{time_stamp}_acc_size_{image_size[0]}_{image_size[1]}.jpeg')