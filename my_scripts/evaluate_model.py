from tensorflow import keras
import tensorflow as tf
import numpy as np 

import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Evaluate the model')
    parser.add_argument('--model-dir', required=True, type=str,
                    help='Where is the model?')
    parser.add_argument('--model-type', required=True, type=str,
                    help='What model is it (NN, CNN, Resnet, EfficientNet)?')

    return parser 

args = get_parser().parse_args()

import os 
if not os.path.isfile(args.model_dir):
    raise ValueError(f'Model directory {args.model_dir} does not exist')

print(f'Loading Model from {args.model_dir}')

model_type = args.model_type
if not os.path.exists(f'../my_scripts/plots/{model_type}'):
    # os.mkdir(f'../my_scripts/plots/{model_type}')
    raise ValueError(f'{model_type} dir does not exist. Have you trained it?')
    
model_dir = args.model_dir# #'../my_scripts/models/EfficientNet/en_dense_74_0.90.h5'
model = keras.models.load_model(model_dir)



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

    # pass 

def main():
    test_ds = load_test_data()
    predictions = model.predict(test_ds)
    test_loss, test_acc = model.evaluate(test_ds)

    one_hot_labels = np.concatenate([y for x,y in test_ds])

    from sklearn import metrics 
    cm = metrics.confusion_matrix(y_true=one_hot_labels.argmax(axis=1), y_pred=predictions.argmax(axis=1))#, labels=target_classes)
    print(cm)

    from matplotlib import pyplot as plt
    import time 
    f = plt.figure(figsize=(10,7))
    ax = f.add_subplot(111)
    plt.title(f'{model_type} Model Confusion Matrix - Loss: {test_loss:.2f} - Test Acc: {test_acc:.2f}')
    metrics.ConfusionMatrixDisplay.from_predictions(y_true=one_hot_labels.argmax(axis=1), y_pred=predictions.argmax(axis=1), display_labels=target_classes,cmap='magma', ax=ax, colorbar=False)
    plt.savefig(f'../my_scripts/plots/{model_type}/{int(time.time())}_confusion_matrix.jpg')


if __name__ == '__main__':
    main()