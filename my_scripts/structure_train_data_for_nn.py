import pathlib 
import os
from tqdm import tqdm 
import tensorflow as tf 

base_dir = pathlib.Path('..')
data_train = base_dir / 'data' / 'train'

target_classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
def make_target_train_classes():
    for target_cls in target_classes:
        if os.path.exists(data_train / target_cls):
            print(f'path {data_train/target_cls} exists')
            print(target_cls, len(os.listdir(data_train / target_cls)))
        else: 
            os.makedirs(data_train / target_cls)
            print(f'path {data_train/target_cls} created')

def calculate_valid_images():
    num_skipped = 0 
    for target_cl in target_classes:
        print(target_cl)
        for file_path in tqdm(os.listdir(data_train / target_cl)):
            try:
                fobj = open(data_train / target_cl / file_path, 'rb')
                is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
            finally:
                fobj.close()
            if is_jfif is None:
                num_skipped += 1
    return num_skipped

def move_images(): 
    # copy files to train folder classes folder 
    for img_file in tqdm(os.listdir(data_train)):
        if img_file.endswith('.jpg'):
            img_path = data_train / img_file
            for target_cls in target_classes:
                if img_file.startswith(target_cls):
                    img_path.rename(data_train / target_cls / img_file)
                    
def structure_image_pipeline():
    print(len(os.listdir(data_train)), os.listdir(data_train)[0])
    print('Making Target classes folders')
    make_target_train_classes()
    print('Calculating valid images')
    calculate_valid_images()
    print('Moving images')
    move_images()

    

if __name__ == '__main__':
    structure_image_pipeline()
