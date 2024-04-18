import torch
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision.io import read_image
from torch import nn
import torch.nn.functional as F

import os
import cv2
from skimage import io
import pandas as pd
from PIL import Image
from sys import argv


class Data:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.train_dataset = datasets.CIFAR10(root="data/", train=True, transform=self.transform, download=True)
        self.test_dataset = datasets.CIFAR10(root="data/", train=False, transform=self.transform_test, download=True)
        self.train_data, self.val_set = torch.utils.data.random_split(self.train_dataset, [45000, 5000])

    def getStat(self, train_data):
        train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
        mean = torch.zeros(3)
        std = torch.zeros(3)

        for X, _ in train_loader:
            for i in range(3):
                mean[i] += X[:, i, :, :].mean()
                std[i] += X[:, i, :, :].std()
        mean.div_(len(train_data))
        std.div_(len(train_data))
        return list(mean.numpy()), list(std.numpy())
    
if __name__ == "__main__":
    data = Data()
    print(data.getStat(data.train_dataset))
    # result is ([0.4914008, 0.482159, 0.44653094], [0.20230275, 0.1994131, 0.2009607])
