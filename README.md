<h1 align="center">Denoising diffusion models for out-of-distribution detection</h1>
<p align="center">
Perform reconstruction-based out-of-distribution detection with DDPMs.
</p>

<p align="center">
  <img width="800" height="300" src="https://user-images.githubusercontent.com/7947315/233470531-df6437d7-e277-4147-96a0-6aa354cf2ef4.svg">
</p>


## Intro

This codebase contains the code to perform unsupervised out-of-distribution detection with diffusion models.
It supports the use of DDPMs as well as Latent Diffusion Models (LDM) for dealing with higher dimensional 2D or 3D data.
It is based on work published in [1] and [2].

[1] [Denoising diffusion models for out-of-distribution detection, CVPR VAND Workshop 2023](https://arxiv.org/abs/2211.07740)

[2] [Unsupervised 3D out-of-distribution detection with latent diffusion models, MICCAI 2023](https://arxiv.org/abs/2307.03777)

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

## Run with DDPM
We'll use the example of FashionMNIST as an in-distribution dataset and [SVHN,CIFAR10, CelebA] as out-of-distribution datasets.
### Download and process datasets
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

