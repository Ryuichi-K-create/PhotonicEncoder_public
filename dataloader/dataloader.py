import torch##
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import os
import numpy as np
from PIL import Image
#--------------------------------------------------------
def load_MNIST_data():
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0,),(1,)),lambda x: x.view(-1)])
    root = os.path.join(os.path.dirname(__file__), 'samples', 'mnist_data')
    mnist_train = datasets.MNIST(root=root,download=True,train=True,transform=transform)
    mnist_test = datasets.MNIST(root=root,download=True,train=False,transform=transform)
    return(mnist_train,mnist_test)
#--------------------------------------------------------
def load_Fmnist_data():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),  # ±10度回転
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # 推奨値
        lambda x: x.view(-1)
    ])

    root = os.path.join(os.path.dirname(__file__), 'samples', 'Fmnist_data')
    fmnist_train = datasets.FashionMNIST(root=root,download=True,train=True,transform=transform)
    fmnist_test = datasets.FashionMNIST(root=root,download=True,train=False,transform=transform)
    return(fmnist_train,fmnist_test)
#--------------------------------------------------------
def load_CIFAR10_data():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                            (0.5, 0.5, 0.5))])
    root = os.path.join(os.path.dirname(__file__), 'samples', 'cifar10_data')
    cifar10_train = datasets.CIFAR10(root=root, download=True, train=True, transform=transform)
    cifar10_test = datasets.CIFAR10(root=root, download=True, train=False, transform=transform)

    return(cifar10_train,cifar10_test)

#--------------------------------------------------------
class NumpyCINIC10Dataset(Dataset):
    def __init__(self, img_path, label_path, transform=None):
        self.images = np.load(img_path)
        self.labels = np.load(label_path)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

def load_CINIC10_data():
    transform = transforms.Compose([
        transforms.Lambda(lambda img: Image.fromarray(img)), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                            (0.5, 0.5, 0.5))])
    root = os.path.join(os.path.dirname(__file__), 'samples', 'cinic10_data','main_data')
    train_dataset = NumpyCINIC10Dataset(img_path=os.path.join(root, 'cinic10_train_imgs.npy'),
                                         label_path=os.path.join(root, 'cinic10_train_labels.npy'),
                                         transform=transform)
    valid_dataset = NumpyCINIC10Dataset(img_path=os.path.join(root, 'cinic10_valid_imgs.npy'),
                                         label_path=os.path.join(root, 'cinic10_valid_labels.npy'),
                                         transform=transform)
    cinic10_test = NumpyCINIC10Dataset(img_path=os.path.join(root, 'cinic10_test_imgs.npy'),
                                        label_path=os.path.join(root, 'cinic10_test_labels.npy'),
                                        transform=transform)

    cinic10_train = torch.utils.data.ConcatDataset([train_dataset, valid_dataset])

    train_size = 50000
    test_size = 10000
    np.random.seed(42)  # For reproducibility
    train_indices = np.random.choice(len(cinic10_train), train_size, replace=False)
    cinic10_train = Subset(cinic10_train, train_indices)

    test_indices = np.random.choice(len(cinic10_test), test_size, replace=False)
    cinic10_test = Subset(cinic10_test, test_indices)

    return(cinic10_train,cinic10_test)
#--------------------------------------------------------

def get_new_dataloader(data_train,data_test,batch_size=64):
    train_dataloader = DataLoader(data_train,batch_size,shuffle=True)
    
    test_dataloader = DataLoader(data_test,batch_size,shuffle=False)
    return train_dataloader, test_dataloader
