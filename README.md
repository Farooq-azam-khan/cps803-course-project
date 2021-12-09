# CPS 803 - Garbage Classification 
- This repo contains all the test code samples, data set configurations and pipeline scripts 

## Prepare the Dataset 

- You can download the datset with the following command in the `data` folder
```bash
kaggle datastes download asdasdasasdas/garbage-classification
```
- move the contents in the `"Garbage Classification/Garbage Classification"` up 1 directory


## Running Models and Data Preparation Steps
- when running scripts change directory in to the `my_scripts` folder first
    - This is because we use relative pathing 
- To prepare the dataset run `python create_train_test_split.py` and `python structure_train_data_for_nn.py`
    - This is assuming you have prepared the dataset in `"data/Garbage Classification"` folder

- To run an efficient net model use: `python efficientnetb[n].py` where `[n]` is either `1`,`2`,`3`,or `4`
