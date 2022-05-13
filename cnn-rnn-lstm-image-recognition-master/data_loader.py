import numpy as np
import torchvision


class DataLoader_:
    def __init__(self):
        self.train_data = np.zeros(1)

    def get_mnist_(self):
        self.train_data = torchvision.datasets.MNIST(
            root='../input/mnist/mnist/',
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )

    def get_train_data(self):
        return self.train_data


#define custom data loader 

from torchvision import transforms
import torch.utils.data as data

class DataLoader:
    def __init__(self):
        self.train_data = np.zeros(1)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def get_data(self):
        self.train_data = torchvision.datasets.ImageFolder(root='Users/samuel/Downloads/2017-Namin-et-al-DeepPheno 2/DeepPhenoData', 
            transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean,
                         std=self.std)
])
        )

    def get_train_data(self):
        return self.train_data




