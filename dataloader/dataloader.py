import torch##
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import os
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#--------------------------------------------------------
def load_Covtype_data():
    test_size = 0.2
    file_path = os.path.join(os.path.dirname(__file__), 'samples',  'covtype.csv')
    data = pd.read_csv(file_path)
    data = data.dropna()

    X = data.drop(columns=['Cover_Type'])
    y_origin = data['Cover_Type']-1

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_origin, test_size=test_size, shuffle=True)

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                                   torch.tensor(y_train.values, dtype=torch.long))
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32), 
                                                  torch.tensor(y_test.values, dtype=torch.long))
    counts = np.bincount(y_origin)
    return train_dataset, test_dataset
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
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(10),  # ±10度回転
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # 推奨値
        lambda x: x.view(-1)
    ])

    root = os.path.join(os.path.dirname(__file__), 'samples', 'Fmnist_data')
    fmnist_train = datasets.FashionMNIST(root=root,download=True,train=True,transform=transform)
    fmnist_test = datasets.FashionMNIST(root=root,download=True,train=False,transform=transform)
    return(fmnist_train,fmnist_test)
#--------------------------------------------------------
def load_Fmnist_data_train(split_train=50000, split_test=10000):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),  # ±10度回転
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # 推奨値
        lambda x: x.view(-1)
    ])

    root = os.path.join(os.path.dirname(__file__), 'samples', 'Fmnist_data')

    # train 全部(60000)を取ってくる
    fmnist_full = datasets.FashionMNIST(root=root, download=True, train=True, transform=transform)

    # 50000 / 10000 に分割
    train_data, val_data = torch.utils.data.random_split(
        fmnist_full, [split_train, split_test], generator=torch.Generator().manual_seed(42)
    )

    return (train_data, val_data)
#--------------------------------------------------------
def load_csv_Fmnist_data(data_id=0):
    file_name = {0: 'fft_magnitude_low.csv',
                 1: 'fft_magnitude_high.csv',
                 2:'fft_phase_low.csv',
                 3:'fft_phase_high.csv'
                 }
    file_path = os.path.join(os.path.dirname(__file__), 'samples',
                             'fashion_mnist_fft',  file_name[data_id])
    data = pd.read_csv(file_path, header=None)
    X = data.iloc[:, 0:32].to_numpy()
    y = data.iloc[:, 33].to_numpy()  # 正しいラベル列は33列目

    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)標準化しない
    X = data.iloc[:, 0:32].to_numpy()    # DataFrame -> numpy array に変換
    y = data.iloc[:, 33].to_numpy()      # Series -> numpy array に変換

    # 標準化しない場合でも numpy 配列にしてから分割する
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=50000, test_size=10000, shuffle=True, random_state=42
    )

    # numpy -> Torch tensor
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.int64))
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test.astype(np.float32)),
        torch.from_numpy(y_test.astype(np.int64))
    )
    return (train_dataset, test_dataset)
#--------------------------------------------------------
def load_compressed_Fmnist_data(data_id=0):
    file_name = {0: 'fashion_mnist_fft_features_inputdata_Norm0.50_Results_Adjust.csv',
                 1: 'fashion_mnist_fft_features_inputdata_Norm0.50_Results_NoAdjust.csv',
                 2:'fashion_mnist_fft_features_inputdata_Norm0.75_Results_Adjust.csv',
                 3:'fashion_mnist_fft_features_inputdata_Norm0.75_Results_NoAdjust.csv',
                 4:'fashion_mnist_fft_features_inputdata_Norm1.00_Results_Adjust.csv',
                 5:'fashion_mnist_fft_features_inputdata_Norm1.00_Results_NoAdjust.csv'
                 }
    file_path = os.path.join(os.path.dirname(__file__), 'samples','fashion_mnist_fft',  file_name[data_id])
    data = pd.read_csv(file_path, header=None)
    X = data.iloc[:, 0:17].to_numpy()
    y = data.iloc[:, 18].to_numpy()  # 正しいラベル列は18列目

    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)標準化しない

    # 標準化しない場合でも numpy 配列にしてから分割する
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=50000, test_size=10000, shuffle=True, random_state=42
    )

    # numpy -> Torch tensor
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.int64))
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test.astype(np.float32)),
        torch.from_numpy(y_test.astype(np.int64))
    )
    return (train_dataset, test_dataset)

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

#--------------------------------------------------------
def get_new_dataloader(data_train,data_test,batch_size=64):
    train_dataloader = DataLoader(data_train,batch_size,shuffle=True)
    
    test_dataloader = DataLoader(data_test,batch_size,shuffle=False)
    return train_dataloader, test_dataloader
#--------------------------------------------------------
