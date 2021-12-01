import tensorflow as tf 
import pathlib 
import matplotlib.pyplot as plt
import itertools
import numpy as np
from matplotlib import style
import tikzplotlib

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


def plot_loss(history, time_stamp, batch_size, image_size, model_type, epochs, save_as_tex=True):
    plt.figure()
    plt.title(f'Simple NN Loss - batch size {batch_size}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(range(1, epochs+1), history.history['val_loss'])
    plt.plot(range(1, epochs+1), history.history['loss'])
    plt.legend(['Validation Loss', 'Training Loss'])
    
    plt.savefig(f'./plots/{model_type}/{time_stamp}_loss_size_{image_size[0]}_{image_size[1]}.jpeg')
    if save_as_tex:
        tikzplotlib.save(f'./plots/{model_type}/{time_stamp}_loss_size_{image_size[0]}_{image_size[1]}.tex')
    plt.close()


def plot_accuracy(history, time_stamp, batch_size, image_size, model_type, epochs, save_as_tex=True):
    plt.figure()
    plt.title(f'Simple NN Accuracy - batch size {batch_size}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(range(1, epochs+1), history.history['val_accuracy'])
    plt.plot(range(1, epochs+1), history.history['accuracy'])
    plt.legend(['Validation Accuracy', 'Training Accuracy'])
    plt.savefig(f'./plots/{model_type}/{time_stamp}_acc_size_{image_size[0]}_{image_size[1]}.jpeg')
    if save_as_tex:
        tikzplotlib.save(f'./plots/{model_type}/{time_stamp}_acc_size_{image_size[0]}_{image_size[1]}.tex')
    plt.close()


def plot_confusion_matrix(cm, test_loss, test_acc, classes, time_stamp, model_type, normalize=False, save_as_tex=True):
    plt.title(f'{model_type} Model Confusion Matrix - Loss: {test_loss:.2f} - Test Acc: {test_acc:.2f}')

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(f'{model_type} Confusion Matrix')
    plt.colorbar()

    if classes is not None:
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel(f'Predicted label\nAccuracy: {test_acc:.2f} - LossS: {test_loss:.2f} - Misclassification: {misclass:.2f}')
    # plt.show()
    plt.savefig(f'./plots/{model_type}/{time_stamp}_confusion_matrix_size.jpeg')
    if save_as_tex:
        tikzplotlib.save(f'./plots/{model_type}/{time_stamp}_confusion_matrix_size.tex')
    plt.close()