<h1 align="center">Flow Matching for out-of-distribution detection</h1>


## Intro

This codebase contains the code to perform unsupervised out-of-distribution detection with flow matching.


## Setup

### Install
Create a fresh virtualenv (this codebase was developed and tested with Python 3.8) and then install the required packages:

```pip install -r requirements.txt```


### Setup paths
Select where you want your data and model outputs stored.
```
data_root=/root/for/downloaded/dataset
output_root=/root/for/saved/models
```

## Run with FM

### Download and process datasets
We use 3 datasets in this project: Fashionmnist, CIFAR10, SVHN. Use the following command to download datasets
```bash
python src/data/get_computer_vision_datasets.py --data_root=${data_root}
```
N.B. If the error shows up for CeleA, just ignore it because we will not use it anyway.

To use your own data, you just need to provide separate csvs containing paths for the train/val/test splits.

### Train models
We use CIFAR10 as an example for training following the command below:

1. for flow matching with stochastic interpolation:
```
python main_SPFM.py
```
2. for vinilla flow matching:
```
python main_FM.py
```   

