import torch
import torch.nn as nn

class MLP_for_10(nn.Module):#10値分類なら使える。
    def __init__(self,potential_dim,num_layer = 2,fc='relu'):
        super(MLP_for_10, self).__init__()
        layers = []
        current_dim = potential_dim
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

    def forward(self, x):
        x = self.model(x)
        return x

