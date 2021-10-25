import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

path = str('/home/anindya/Documents/pytorch/torch_utils' 'sample_data')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_path = os.path.join(path, 'cifar_train')
test_path = os.path.join(path, 'cifar_test')

class Data(Dataset):
    def __init__(self, data, transform):
        super(Data, self).__init__()
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index]
        if self.transform:
            image = self.transform(image)
        return image, label

train_dataset = Data(CIFAR10(root=train_path, download=True, train=True), ToTensor())
test_dataset =  Data(CIFAR10(root=test_path, download=True, train=False), ToTensor())

train_data = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_data = DataLoader(test_dataset, batch_size=64)

model = ModelZoo('vgg16', pretrained=True, num_classes=10).create_model()

for (data, label) in train_data:
    X = data.to(device) 
    y = label.to(device)
    preds = model(X)
    print(preds.shape)
    break