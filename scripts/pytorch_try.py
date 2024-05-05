# -*- coding: utf-8 -*-
"""
Created on Sun May  5 13:23:29 2024

@author: matus
"""
#%%
# Pytorch modules
import torch
# main class used for making neural networks
from torch import nn
# pytorch has all the datasets I need already in its computer vision module
from torchvision import datasets
# used to parallelize loading (my laptop has only 4 cores)
from torch.utils.data import DataLoader
# used to transform input data into desired format
from torchvision import transforms

# general python modules
import os
import numpy as np
from matplotlib import pyplot as plt

print('all modules imported!')

#%%
# define transformations we will apply to images before using them
transform = transforms.Compose([
    # same size all (MNIST images 28x28)
    transforms.Resize([28,28]),
    # transform all to greyscale
    transforms.Grayscale(),
    # transform to tensor representation
    transforms.ToTensor()])

# get the MNIST data and apply our transform to it and save to same directory
mnist = datasets.MNIST(root='.', # same directory
                       download=True, 
                       transform = transform)

#%%
# batch size seems to be a parameter that one can play with a lot
# will try different ones (apparently smaller batch size = faster training but more noise)

# iterator of the dataset (always returns the n = batch size of 
# images with labels in each iteration)
dataloader = DataLoader(dataset = mnist, 
                        batch_size = 64,
                        num_workers = 4) # 4 cpu cores on my laptop

image, label = next(iter(dataloader))

print(label)
