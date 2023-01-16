'''
Transforms can be applied to PIL images, tensors, ndarrays, or custom data
during creation of the DataSet
complete list of built-in transforms: 
https://pytorch.org/docs/stable/torchvision/transforms.html
On Images
---------
CenterCrop, Grayscale, Pad, RandomAffine
RandomCrop, RandomHorizontalFlip, RandomRotation
Resize, Scale
On Tensors
----------
LinearTransformation, Normalize, RandomErasing
Conversion
----------
ToPILImage: from tensor or ndrarray
ToTensor : from numpy.ndarray or PILImage
Generic
-------
Use Lambda 
Custom
------
Write own class
Compose multiple Transforms
---------------------------
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])
'''

import torch
import numpy as np
import torchvision
import math
from torch.utils.data import Dataset, DataLoader
#https://github.com/patrickloeber/pytorchTutorial/blob/69637721ed9fa2ccf81a1093a928a52bbb591439/data/wine/wine.csv

class WineDataset(Dataset):
    def __init__(self, transform=None):
        # data loading
        xy = np.loadtxt('./data/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        # self.x = torch.from_numpy(xy[:, 1:])
        # self.y = torch.from_numpy(xy[:, [0]])
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]

        self.num_samples = xy.shape[0]
        self.transform = transform
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample
    def __len__(self):
        return self.num_samples
    
dataset = WineDataset()

# Load whole dataset with DataLoader
# shuffle: shuffle data, good for training
# num_workers: faster loading with multiple subprocesses
# !!! IF YOU GET AN ERROR DURING LOADING, SET num_workers TO 0 !!!
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)
# get first sample and unpack
first_data = dataset[0]
features, labels = first_data

# convert to an iterator and look at one random sample
dataiter = iter(dataloader)
# print(dataset)
data = dataiter.next()
features, labels = data
# print(features, labels)

num_epochs = 2

total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
# print(total_samples, n_iterations)


for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        #forward backward pass, update
        if (i+1) % 5 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}, Step {i+1}/{n_iterations}| Inputs {inputs.shape} | Labels {labels.shape}')
# transform=torchvision.transforms.ToTensor(), download=True)
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
# transform=torchvision.transforms.ToTensor(download=True)


class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform:
    def __init__(self, factor):
        self.factor = factor
    def __call__(self, sample):
        inputs, targets = sample
        return inputs*self.factor, targets


composed = torchvision.transforms.Compose([ToTensor(), MulTransform(4)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]

features, labels = first_data

print(labels)
