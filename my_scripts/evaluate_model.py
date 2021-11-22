from tensorflow import keras
import tensorflow as tf
import numpy as np 

model_dir = '../my_scripts/models/EfficientNet/simple_909_0.90.h5'
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
test_ds = load_test_data()
predictions = model.predict(test_ds)
one_hot_labels = np.concatenate([y for x,y in test_ds])

from sklearn import metrics 
cm = metrics.confusion_matrix(y_true=one_hot_labels.argmax(axis=1), y_pred=predictions.argmax(axis=1))#, labels=target_classes)
print(cm)

from matplotlib import pyplot as plt
import time 
f = plt.figure(figsize=(10,7))
ax = f.add_subplot(111)
plt.title('Efficient Net B0 Model Confusion Matrix')
metrics.ConfusionMatrixDisplay.from_predictions(y_true=one_hot_labels.argmax(axis=1), y_pred=predictions.argmax(axis=1), display_labels=target_classes,cmap='magma', ax=ax, colorbar=False)
plt.savefig(f'../my_scripts/plots/EfficientNet/confusion_matrix_{int(time.time())}.jpg')