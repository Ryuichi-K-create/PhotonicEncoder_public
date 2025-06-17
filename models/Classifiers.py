import torch
import torch.nn as nn
import numpy as np

class MLP_for_10(nn.Module):#10値分類なら使える。
    def __init__(self,potential_dim,num_layer = 2,fc='relu',n_patches=64):
        super(MLP_for_10, self).__init__()
        layers = []
        current_dim = potential_dim

        self.bn = nn.BatchNorm1d(potential_dim//n_patches)
        func ={
            'relu':nn.ReLU(),
            'tanh':nn.Tanh(),
            'leakyrelu':nn.LeakyReLU(),
            'sigmoid':nn.Sigmoid()
        }
        ac_func = func[fc]
        for i in range(num_layer-1):
            next_dim = max(10,current_dim//2)
            layers.append(nn.Linear(current_dim,next_dim))
            layers.append(ac_func)
            current_dim = next_dim
        layers.append(nn.Linear(current_dim,10))
        self.model = nn.Sequential(*layers)

    def forward(self, x,b):
        x = self.bn(x)
        x = x.reshape(b, -1)
        x = self.model(x)
        return x


class CNN_for10(nn.Module):
    def __init__(self,potential_dim,num_layer = 2,fc='relu',n_patches=64):
        super(CNN_for10, self).__init__()
        feat_dim = potential_dim // n_patches
        self.bn = nn.BatchNorm2d(feat_dim)
        self.side = int(np.sqrt(n_patches))
        self.conv1 = nn.Conv2d(feat_dim,32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32,64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * (self.side // 4) * (self.side // 4), 256)
        self.fc2 = nn.Linear(256, 10)
        func ={
            'relu':nn.ReLU(),
            'tanh':nn.Tanh(),
            'leakyrelu':nn.LeakyReLU(),
            'sigmoid':nn.Sigmoid()
        }
        self.func = func[fc]

    def forward(self, x,b):
        x = x.view(b, self.side, self.side, -1).permute(0, 3, 1, 2)
        x = self.bn(x)
        x = self.conv1(x)
        x = self.func(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.func(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.func(self.fc1(x))
        x = self.fc2(x)
        return x
    

