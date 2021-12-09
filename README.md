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

- To run an efficient net model run: `python efficientnetb[n].py` where `[n]` is either `1`,`2`,`3`,or `4`
- To run the resnet models run: `python resnet.py`
- To run the keras tuner run: `python tune_script.py`
    - if you are running it on a server and would like it to run in the background run this: `nohup python tune_script.py > run_tuner.txt &`
    - to check up on the process you can `cat run_tuner.txt` or `ps ax | grep tune_script.py`
