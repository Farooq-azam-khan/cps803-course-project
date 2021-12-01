import tensorflow as tf 
import pathlib 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
import tikzplotlib

from sklearn import metrics 

style.use('ggplot')

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


def plot_loss(history, experiment_path: pathlib.Path, batch_size, image_size, model_type, epochs, save_as_tex=True):
    plt.figure()
    plt.title(f'Simple NN Loss - batch size {batch_size}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(range(1, epochs+1), history.history['val_loss'])
    plt.plot(range(1, epochs+1), history.history['loss'])
    plt.legend(['Validation Loss', 'Training Loss'])
    
    plt.savefig(experiment_path / f'loss_size_{image_size[0]}_{image_size[1]}.jpeg')
    if save_as_tex:
        tikzplotlib.save(experiment_path / f'loss_size_{image_size[0]}_{image_size[1]}.tex')
    plt.close()


def plot_accuracy(history, experiment_path: pathlib.Path, batch_size, image_size, model_type, epochs, save_as_tex=True):
    plt.figure()
    plt.title(f'Simple NN Accuracy - batch size {batch_size}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(range(1, epochs+1), history.history['val_accuracy'])
    plt.plot(range(1, epochs+1), history.history['accuracy'])
    plt.legend(['Validation Accuracy', 'Training Accuracy'])
    plt.savefig(experiment_path / f'acc_size_{image_size[0]}_{image_size[1]}.jpeg')
    if save_as_tex:
        tikzplotlib.save(experiment_path / f'acc_size_{image_size[0]}_{image_size[1]}.tex')
    plt.close()


def plot_confusion_matrix(one_hot_labels, predictions, test_loss, test_acc, target_classes, target_path, model_type, save_as_tex=True):
    style.use('classic')
    plt.title(f'{model_type} Model Confusion Matrix - Loss: {test_loss:.2f} - Test Acc: {test_acc:.2f}')

    f = plt.figure(figsize=(10,7))
    ax = f.add_subplot(111)
    plt.title(f'{model_type} Model Confusion Matrix - Loss: {test_loss:.2f} - Test Acc: {test_acc:.2f}')
    metrics.ConfusionMatrixDisplay.from_predictions(y_true=one_hot_labels.argmax(axis=1), y_pred=predictions.argmax(axis=1), display_labels=target_classes,cmap='magma', ax=ax, colorbar=False)
    plt.savefig(target_path / 'confusion_matrix.jpeg')

    if save_as_tex:
        pass 
        # code below does not work 
        # tikzplotlib.save((target_path / 'confusion_matrix.tex'))
    plt.close()
    style.use('ggplot')

def make_experiment_dir(model_type, time_stamp):
    import pathlib
    import os
    new_path = pathlib.Path('experiments') / model_type / str(time_stamp)
    if not new_path.exists():
        os.makedirs(new_path)
        os.makedirs(new_path / 'models')
        os.makedirs(new_path / 'plots')
    return new_path
    
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

def save_history(history, experiment_path: pathlib.Path):
    np.save(experiment_path / 'history.npy', history)