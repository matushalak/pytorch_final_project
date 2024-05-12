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
                        num_workers = 0) # 4 cpu cores on my laptop

image, label = next(iter(dataloader))

print('One batch:', image.shape)

#%%
from numpy.random import randint
fig, axes = plt.subplots(3,3,figsize = (5,5))
axes = axes.flatten()
for ax, image_eg in enumerate(randint(0,64,size = 9)):
    axes[ax].set_title(f'Label: {label[image_eg]}')
    axes[ax].axis('off')
    axes[ax].imshow(transforms.ToPILImage()(image[image_eg]))
# Example data
plt.show()

#%%
# fully connected (fc) block (will repeat for multiple hidden layers)
def fc_block(input_size:int,
            output_size:int,
            activation_function:object):
    return nn.Sequential(
        # how to perform "weighed sum" on 'output' of previous layer
        nn.Linear(input_size,output_size),
        # what to do with "weighed sum"
        activation_function())
    
# start with fully connected network
class My_First_NN (nn.Module):
    def __init__(self,
                 in_channels:int,
                 hidden_layers_sizes:list,
                 out_channels:int,
                activation:object):
        # initialize whatever is in __init__ of nn.Module
        super(My_First_NN, self).__init__()
        # this one way how input & hidden layers can be defined
        # sizes (number of nodes) of each layer
        layer_sizes = [in_channels,*hidden_layers_sizes]
        # make blocks of fully connected layers 
        fully_connected_blocks = [fc_block(in_n, out_n, activation)
                                  for in_n, out_n in zip(layer_sizes, layer_sizes[1:])]
        # these will be sequentially executed in the forward pass
        self.layers = nn.Sequential(*fully_connected_blocks,
                                    # for output layer only want weighed sum
                                    nn.Linear(layer_sizes[-1],out_channels))
        
    def forward(self, x):
        # flatten input
        try:
            x = x.view(x.size(0),-1)
        except TypeError:
            breakpoint()
        # forward pass of input through network
        x = self.layers(x)
        # at the end, x is a list of 10 numbers from 0-10 telling us
        # which label the model predicts
        return x

first_model = My_First_NN(784,
                          # completely arbitrary number of hidden layers & nodes
                         [300,200,300],
                         10,
                         nn.ReLU)
# inspect model architecture
print(first_model) # yes, as intended
#%%
# training loop
def train_network (network:nn.Module,
                   data_loader:torch.utils.data.DataLoader,
                   epochs:int,
                   # these seem to be popular, beyond me why exactly
                   loss_function:nn.Module = nn.CrossEntropyLoss(),
                   # smaller so that model won't overfit
                   learning_rate:float = 0.001,
                   # here I found SDG and Adam, will use SGD because I undersatnd it better
                   optimizer:torch.optim.Optimizer = torch.optim.SGD):
    # set NN in training mode
    network.train(True)

    # set up optimizer
    optimizer = optimizer(network.parameters(), lr = learning_rate)
    # go through epochs
    for e,epoch in enumerate(range(epochs)):
        # load a batch of training data for given epoch
        for b,batch in enumerate(data_loader):
            samples, labels = batch[0], batch[1]
            # reset gradient, need to calculate gradient afresh every time
            optimizer.zero_grad()
            # run input data through network
            predictions = network.forward(samples)
            # calculate eror
            error = loss_function(predictions, labels)
            # calculate gradient
            error.backward()
            # update weights by stepping down the gradient by learning rate sized step
            optimizer.step()
            print(e,b, 'done')

# let's train, see how long it takes
train_network(network=first_model,
             data_loader=dataloader,
             epochs = 5)

#%%
# testing loop
def test_performance (loader:torch.utils.data.DataLoader,
                      network:nn.Module):
    # overall performance
    ncorrect, nclassification = 0,0

    # set model into testing mode
    network.eval()
    # context manager disables gradient calculation (only evaluation, doesn't change weights)
    with torch.no_grad():
        # don't know how to quickly and elegantly solve this
        # for my handdrawn images
        if isinstance(loader, list):
            images, labels = loader
            outputs = network(images)
            # get the index at which the maximum probability occured (reduce over columns = dimension 1{dimension 0 is rows})
            predictions = torch.argmax(outputs,1)
            # makes a mask of True & False, summing over true gives number correct since True = 1
            ncorrect += (predictions == labels).sum()
            nclassification += len(predictions)
        else:
            # normal
            for images, labels in loader:
                outputs = network(images)
                # get the index at which the maximum probability occured (reduce over columns = dimension 1{dimension 0 is rows})
                predictions = torch.argmax(outputs,1)
                # makes a mask of True & False, summing over true gives number correct since True = 1
                ncorrect += sum(predictions == labels)
                nclassification += len(predictions)
                    
    performance = float(ncorrect)/float(nclassification)
    print(f'{ncorrect} / {nclassification} samples correct -> accuracy {performance*100:.2f} %')
    return performance
          
# test on training data
performance = test_performance(dataloader,
                              first_model)

#%%
# for working with our images for later
from PIL import Image, ImageOps
from numpy import array, dstack
# filenames containing labels
my_testing_set = os.listdir('drawing')
# extract labels
labels = [int(x[0]) for x in my_testing_set]
# extract data
data = []
for sample in my_testing_set:
    # get image
    sample = ImageOps.grayscale(Image.open(f'drawing/{sample}'))
    sample = sample.resize((28,28))
    # turn into array
    my_array = np.array(sample).astype('float32')/255
    #print(my_array.shape)
    data.append(my_array)
# transform to tensor shape like before
data = dstack(data).reshape((len(my_testing_set),1,28,-1))
print(data.shape)

my_loader = [data,labels]

# handdrawn data performance
hand_drawn_performance = test_performance(my_loader,

                                          first_model)

#%%
import numpy as np
models = ['fully_connected_0.0734.pt',
          'fully_connected_0.6562.pt',
          'fully_connected_0.4943.pt',
          'fully_connected_0.1234.pt']
model_name = np.argmax([float(m.split('_')[-1].split('.pt')[0]) for m in models])
print(models[model_name])
#fc_model = fc_model.load_state_dict(torch.load('.' + f"\{models[fc_model]}") 
