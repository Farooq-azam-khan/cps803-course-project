import os 
import pathlib 
import shutil 
from sklearn.model_selection import train_test_split
import numpy as np

def main():
    base = pathlib.Path('../data/Garbage classification')
    classifications =  os.listdir(base)

    if not os.path.exists('../data/train'):
        os.mkdir('../data/train')
    if not os.path.exists('../data/test'):
        os.mkdir('../data/test')
        
    for classification in classifications:
        image_dir = np.array(os.listdir(base / classification))
        y_label = np.array([classification]*len(image_dir))
        X_train, X_test, _, _ = train_test_split(image_dir, y_label, test_size=0.3, random_state=42)
        print(f'{classification}: {len(image_dir)} = {len(X_train)} (train) + {len(X_test)} (test)')
        print('moving files to trainning folder...')
        for img_file in X_train:
            shutil.copy(base/classification/img_file, pathlib.Path(f'../data/train/{img_file}'))
        
        
        print('moving files to testing folder...')
        for img_file in X_test:
            shutil.copy(base/classification/img_file, pathlib.Path(f'../data/test/{img_file}'))

if __name__ == '__main__':
    main()
    # Tests 
    assert 121 == len(list(pathlib.Path('../data/test/').glob('cardboard*')))
    assert 151 == len(list(pathlib.Path('../data/test/').glob('glass*')))
    assert 123 == len(list(pathlib.Path('../data/test/').glob('metal*')))
    assert 179 == len(list(pathlib.Path('../data/test/').glob('paper*')))
    assert 145 == len(list(pathlib.Path('../data/test/').glob('plastic*')))
    assert 42 == len(list(pathlib.Path('../data/test/').glob('trash*')))

    assert 282 == len(list(pathlib.Path('../data/train/').glob('cardboard*')))
    assert 350 == len(list(pathlib.Path('../data/train/').glob('glass*')))
    assert 287 == len(list(pathlib.Path('../data/train/').glob('metal*')))
    assert 415 == len(list(pathlib.Path('../data/train/').glob('paper*')))
    assert 337 == len(list(pathlib.Path('../data/train/').glob('plastic*')))
    assert 95 == len(list(pathlib.Path('../data/train/').glob('trash*')))