import torch
import torch.nn as nn

from .IntegrationModel import PMEncoder, IMEncoder, MZMEncoder, LIEncoder

class Cell(nn.Module):
    def __init__(self, x_dim, z_dim,enc_type):
        super().__init__()
        encoders = {
            'PM':PMEncoder,
            'IM':IMEncoder,
            'MZM':MZMEncoder,
            'LI':LIEncoder
        }
        self.enc1 = encoders[enc_type](x_dim+z_dim,z_dim)
        self.fc1 = nn.Linear(z_dim,z_dim)
        self.bn = nn.BatchNorm1d(z_dim)
        self.act = nn.ReLU()
    def forward(self,zx):
        z = self.enc1(zx)
        #以下、積和演算電子回路-----------------------
        z = self.bn(z)
        z = self.fc1(z)
        z = self.act(z)
        return z