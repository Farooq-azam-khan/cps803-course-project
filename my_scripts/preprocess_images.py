from matplotlib import image 
from tqdm import tqdm 
import numpy as np

import pathlib 


label_to_int_mapping = {
    'cardboard': 1, 
    'glass': 2, 
    'metal': 3, 
    'paper': 4, 
    'plastic': 5, 
    'trash': 6
}

int_to_label_mapping = {
    '1': 'cardboard', 
    '2': 'glass', 
    '3': 'metal', 
    '4': 'paper', 
    '5': 'plastic',
    '6': 'trash'
}

def load_image_files(train_data_path, test_data_path, assertion=False):
    train_data_files = list(train_data_path.glob('*.jpg'))
    test_data_files = list(test_data_path.glob('*.jpg'))
    if assertion:
        assert 1766 == len(train_data_files)
        assert 761 == len(test_data_files)
    return train_data_files, test_data_files

def flatten_and_normalize(img):
    return img.flatten() / 255.0

def grey_scale_image(img):
    return np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])#img.mean(axis=2)

def preprocess_data(image_files, col_dim, use_grey_scale):
    
   
    X = np.zeros((len(image_files),col_dim))
    y = np.zeros((len(image_files),))
    for example_idx, image_file in enumerate(image_files):
        if use_grey_scale: 
            img_data = flatten_and_normalize(grey_scale_image(image.imread(image_file)))
        else:
            img_data = flatten_and_normalize(image.imread(image_file))
        X[example_idx] = img_data 
        if 'cardboard' in str(image_file):
            y[example_idx] = label_to_int_mapping['cardboard']
        elif 'glass' in str(image_file):
            y[example_idx] = label_to_int_mapping['glass']
        elif 'metal' in str(image_file):
            y[example_idx] = label_to_int_mapping['metal']
        elif 'paper' in str(image_file):
            y[example_idx] = label_to_int_mapping['paper']
        elif 'plastic' in str(image_file):
            y[example_idx] = label_to_int_mapping['plastic']
        else: 
            y[example_idx] = label_to_int_mapping['trash']
    return X, y
    
def get_preprocessed_train_test_images(use_grey_scale=True, assertion=False):
    base = pathlib.Path('../data')
    train_data_path = base / 'train'
    test_data_path = base / 'test'
    train_data_files, test_data_files = load_image_files(train_data_path, test_data_path)
    img0 = image.imread(train_data_files[0])
    orig_size = image.imread(train_data_files[0]).shape

    if assertion: 
        for train_data_file in tqdm(train_data_files):
            assert orig_size == image.imread(train_data_file).shape
            
        for train_data_file in tqdm(train_data_files):
            assert orig_size == image.imread(train_data_file).shape

    if use_grey_scale: 
        col_dim = grey_scale_image(img0).flatten().shape[0]
    else: 
        col_dim = img0.flatten().shape[0]
    print(grey_scale_image(img0).shape)
    X_train, y_train = preprocess_data(train_data_files, col_dim, use_grey_scale)
    X_test, y_test = preprocess_data(test_data_files, col_dim, use_grey_scale)
    return X_train, X_test, y_train, y_test