import torch##
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


def load_MNIST_data():
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,)),lambda x: x.view(-1)])
    root = os.path.join(os.path.dirname(__file__), 'samples', 'mnist_data')
    mnist_train = datasets.MNIST(root=root,download=True,train=True,transform=transform)
    mnist_test = datasets.MNIST(root=root,download=True,train=False,transform=transform)
    return(mnist_train,mnist_test)

def get_new_dataloader(data_train,data_test,batch_size=64):
    train_dataloader = DataLoader(data_train,batch_size,shuffle=True)
    test_dataloader = DataLoader(data_test,batch_size,shuffle=False)
    return train_dataloader, test_dataloader
