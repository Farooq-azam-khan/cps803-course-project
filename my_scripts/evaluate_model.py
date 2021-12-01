import os 
import argparse
import pathlib

import numpy as np 
from sklearn import metrics 

from tensorflow import keras
import tensorflow as tf
import time 
import pathlib 
from helpers import plot_confusion_matrix

def get_parser():
    parser = argparse.ArgumentParser(description='Evaluate the model')
    parser.add_argument('--model-dir', required=True, type=str,
                    help='Where is the model?')
    parser.add_argument('--model-type', required=True, type=str,
                    help='What model is it (NN, CNN, Resnet, EfficientNet)?')

    return parser 



# load test Data 
IMG_PIXELS = 224
image_size = (IMG_PIXELS, IMG_PIXELS)
target_classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def load_test_data():

    print('Loading test data')
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        '../data/test',
        validation_split=0,
        label_mode="categorical",
        seed=1337,
        image_size=image_size,
        batch_size=100,
    )
    return test_ds

def evaluate_model_on_test_data(model, model_type: str, target_path: pathlib.Path):
    test_ds = load_test_data()
    predictions = model.predict(test_ds)
    test_loss, test_acc = model.evaluate(test_ds)

    one_hot_labels = np.concatenate([y for x,y in test_ds])

    cm = metrics.confusion_matrix(y_true=one_hot_labels.argmax(axis=1), y_pred=predictions.argmax(axis=1))#, labels=target_classes)
    print(cm)
    np.save(target_path / 'confusion_matrix.npy', cm)
    
    # plot confusion matrix
    plot_confusion_matrix(
            one_hot_labels=one_hot_labels, 
            predictions=predictions, 
            test_loss=test_loss, 
            test_acc=test_acc, 
            target_classes=target_classes, 
            target_path=target_path,
            model_type=model_type,
            save_as_tex=True)

def main():
    
    args = get_parser().parse_args()

    if not os.path.isfile(args.model_dir):
        raise ValueError(f'Model directory {args.model_dir} does not exist')

    print(f'Loading Model from {args.model_dir}')

    model_type = args.model_type
    if not os.path.exists(f'../my_scripts/plots/{model_type}'):
        # os.mkdir(f'../my_scripts/plots/{model_type}')
        raise ValueError(f'{model_type} dir does not exist. Have you trained it?')
        
    model_dir = args.model_dir# #'../my_scripts/models/EfficientNet/en_dense_74_0.90.h5'
    model = keras.models.load_model(model_dir)
    
    target_path = pathlib.Path('experiments') / model_type / str(int(time.time()))
    evaluate_model_on_test_data(model, model_type, target_path)


if __name__ == '__main__':
    main()