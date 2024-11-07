import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np


NUM_CLASSES = 10

def CIFAR10(batch_size):

    dataset_train = datasets.CIFAR10(
        root='data',
        train=True,
        download=True,
        transform=ToTensor(),
    )
    dataset_valid = datasets.CIFAR10(
        root='data',
        train=False,
        download=True,
        transform=ToTensor(),
    )
    train_loader = DataLoader(
        dataset_train, 
        batch_size=batch_size,
        shuffle=True
    )
    valid_loader = DataLoader(
        dataset_valid, 
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, valid_loader

class AlexNetBaseline(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNetBaseline, self).__init__()
        

        self.c1 = nn.Conv2d(3,64,kernel_size=3,stride=2,padding=1)
        self.r1 = nn.ReLU()
        self.m1 = nn.MaxPool2d(kernel_size=2)
        self.c2 = nn.Conv2d(64,192,kernel_size=3,stride=1,padding=1)
        self.r2 = nn.ReLU()
        self.m2 = nn.MaxPool2d(kernel_size=2)
        self.c3 = nn.Conv2d(192,384,kernel_size=3, stride=1, padding=1)
        self.r3 = nn.ReLU()
        self.c4 = nn.Conv2d(384,256,kernel_size=3, padding=1)
        self.r4 = nn.ReLU()
        self.c5 = nn.Conv2d(256,256,kernel_size=3, padding=1)
        self.r5 = nn.ReLU()
        self.m5 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(256*2*2,4096)
        self.r6 = nn.ReLU()
        self.fc2 = nn.Linear(4096,4096)
        self.r7 = nn.ReLU()
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        ### Forward pass
        y1 = self.c1(x)

        y2 = self.r1(y1)
        y3 = self.m1(y2)

        y4 = self.c2(y3)

        y5 = self.r2(y4)
        y6 = self.m2(y5)

        y7 = self.c3(y6)

        y8 = self.r3(y7)

        y9 = self.c4(y8)

        y10= self.r4(y9)

        y11= self.c5(y10)

        y12= self.r5(y11)
        y13= self.m5(y12)

        y13_size = y13.shape

        y14 = torch.reshape(y13, (-1, 256 * 2 * 2))
        
        y15= self.fc1(y14)

        y16= self.r6(y15)

        y17= self.fc2(y16)

        y18= self.r7(y17)
        
        y19= self.fc3(y18)

        
        return y19
