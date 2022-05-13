import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np


class DataLoader:
    @staticmethod
    def get_train_data(batch_size):
        dataset= torchvision.datasets.ImageFolder(root='/Users/samuel/SP22/CS444/Project2/Plant_Phenotyping_Datasets/Plant/Ara2013-RPi')
        return torch.utils.data.DataLoader(
           dataset,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ],
            batch_size=batch_size,
            shuffle=True))

    @staticmethod
    def get_test_data(test_batch_size):
        dataset = torchvision.datasets.ImageFolder(root='Users/samuel/Downloads/2017-Namin-et-al-DeepPheno 2/DeepPhenoData') 
        return torch.utils.data.DataLoader(dataset,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ],
            batch_size=test_batch_size,
            shuffle=True))

