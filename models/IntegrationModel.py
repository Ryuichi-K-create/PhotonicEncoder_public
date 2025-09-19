import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from .OtherModels import Cell,Cell_fft,DEQFixedPoint,anderson,FFTLowFreqSelector
#--------------------------------------------------------------------
def split_into_kernels(image, kernel_size):
    b, c, h, w = image.shape
    assert (h % kernel_size == 0) and (w % kernel_size == 0), "Image size must be divisible by kernel size"
    patches = image.unfold(2, kernel_size, kernel_size).unfold(3, kernel_size, kernel_size)
    patches = patches.contiguous().view(b, c, -1, kernel_size, kernel_size)
    patches = patches.permute(0, 2, 1, 3, 4)
    return patches      #(b, n_patches, c, kernel_size, kernel_size)
#--------------------------------------------------------------------
class PMEncoder(nn.Module):
    def __init__(self,input_dim,output_dim,alpha,device='cpu'):
        super(PMEncoder,self).__init__()
        phase = torch.rand(output_dim, input_dim) * 2 * np.pi - np.pi
        modulus = torch.ones(output_dim, input_dim)/np.sqrt(input_dim)

        real_part = modulus * torch.cos(phase)
        imag_part = modulus * torch.sin(phase)

        self.B = torch.complex(real_part, imag_part).detach().to(device)
        self.B.requires_grad = False
        self.alpha = (torch.rand(input_dim) - 0.5) * (2*alpha) 
        self.alpha = self.alpha.detach().to(device) 
        self.alpha.requires_grad = False

    def forward(self, x):
        x = torch.exp(1j * self.alpha * x) 
        x = x.T  
        x = torch.matmul(self.B, x).T 
        x = torch.abs(x)**2 
        return x
    
class IMEncoder(nn.Module):
    def __init__(self,input_dim,output_dim,alpha,device='cpu'):
        super(IMEncoder,self).__init__()
        self.B = nn.Parameter(torch.randn(output_dim, 
                                          input_dim) * (1/np.sqrt(input_dim))).detach().to(device)
        self.B.requires_grad = False

    def forward(self, x):
        x = x.T  
        x = torch.matmul(self.B, x).T 
        x = torch.abs(x)**2 
        return x

class MZMEncoder(nn.Module):
    def __init__(self,input_dim,output_dim,alpha,device='cpu'):
        super(MZMEncoder,self).__init__()
        phase = torch.rand(output_dim, input_dim) * 2 * np.pi - np.pi
        modulus = torch.ones(output_dim, input_dim)

        real_part = modulus * torch.cos(phase)
        imag_part = modulus * torch.sin(phase)

        self.B = torch.complex(real_part, imag_part).detach().to(device) 
        self.B.requires_grad = False
        self.alpha = (torch.rand(input_dim) - 0.5) * (2*alpha) 
        self.alpha.requires_grad = False

    def forward(self, x):
        x = 1+torch.exp(1j * self.alpha * x) 
        x = x.T  
        x = torch.matmul(self.B, x).T 
        x = torch.abs(x)**2 
        return x

class LIEncoder(nn.Module):
    def __init__(self,input_dim,output_dim,alpha,device='cpu'):
        super(LIEncoder,self).__init__()
        self.B = nn.Parameter(torch.randn(output_dim, 
                                          input_dim) * (input_dim)).detach().to(device)
        self.B.requires_grad = False

    def forward(self, x):
        x = x.T  
        x = torch.matmul(self.B, x).T  
        return x

#--------------------------------------------------------------------
from .Classifiers import MLP_for_10, CNN_for10, MLP_for_7
#--------------------------------------------------------------------

class Image10Classifier(nn.Module):#10クラスの画像用
    def __init__(self, dataset,kernel_size,leverage,
                 enc_type,alpha,cls_type,num_layer,fc,dropout,device):
        super(Image10Classifier, self).__init__()
        dataset_config = {
            'mnist':     {'img_size': 28, 'channels': 1},
            'cifar-10':  {'img_size': 32, 'channels': 3},
            'fashion-mnist': {'img_size': 28, 'channels': 1},
            'cifar-100': {'img_size': 32, 'channels': 3},
            'cinic-10': {'img_size':32, 'channels':3}
        }
        if dataset not in dataset_config:
            raise ValueError(f"Unknown dataset: {dataset}")
        self.img_size = dataset_config[dataset]['img_size']
        self.channels = dataset_config[dataset]['channels']

        self.kernel_size = kernel_size
        kernel_in = self.channels*kernel_size**2
        feat_dim = int(kernel_in/leverage)
        encoders = {
            'PM':PMEncoder,
            'IM':IMEncoder,
            'MZM':MZMEncoder,
            'LI':LIEncoder,
            'none':PMEncoder
        }
        classifiers = {
            'MLP':MLP_for_10,
            'CNN':CNN_for10
        }

        self.num_patches = (self.img_size//kernel_size)**2
        potential_dim = self.num_patches * feat_dim
        self.split = split_into_kernels 
        self.enc_type = enc_type
        self.encoder = encoders[enc_type](kernel_in,feat_dim,alpha,device) 
        self.classifier =  classifiers[cls_type](potential_dim,num_layer,fc,self.num_patches,dropout).to(device)

    def forward(self, x):
        b=x.size(0)
        x = x.view(b, self.channels, self.img_size, self.img_size)
        x = self.split(x, self.kernel_size)#(b, p, c, k, k)
        x = x.reshape(b * self.num_patches,
                      self.channels * self.kernel_size**2)
        if self.enc_type != 'none':
            x = self.encoder(x) 
        x = self.classifier(x,b)
        return x
#--------------------------------------------------------------------
class Image10Classifier_FFT(nn.Module):#10クラスの画像用(FFT特徴量版)
    def __init__(self, dataset,kernel_size,leverage,
                 enc_type,alpha,cls_type,num_layer,fc,dropout,device):
        super(Image10Classifier_FFT, self).__init__()
        dataset_config = {
            'mnist':     {'img_size': 28, 'channels': 1},
            'cifar-10':  {'img_size': 32, 'channels': 3},
            'fashion-mnist': {'img_size': 28, 'channels': 1},
            'cinic-10': {'img_size':32, 'channels':3}
        }
        if dataset not in dataset_config:
            raise ValueError(f"Unknown dataset: {dataset}")
        self.img_size = dataset_config[dataset]['img_size']
        self.channels = dataset_config[dataset]['channels']

        encoders = {
            'PM':PMEncoder,
            'IM':IMEncoder,
            'MZM':MZMEncoder,
            'LI':LIEncoder,
            'none':PMEncoder
        }
        classifiers = {
            'MLP':MLP_for_10,
            'CNN':CNN_for10
        }
        self.fft_dim = 25 #fft特徴量の次元数
        feat_dim = 17
        self.enc_type = enc_type
        self.fft = FFTLowFreqSelector(out_dim=self.fft_dim, log_magnitude=True)
        self.bn = nn.BatchNorm1d(self.fft_dim).to(device)
        self.ln = nn.LayerNorm(self.fft_dim).to(device)
        self.encoder = encoders[enc_type](self.fft_dim,feat_dim,alpha,device) 
        if enc_type == 'none':
            self.classifier =  classifiers[cls_type](self.fft_dim,num_layer,fc,n_patches=None,dropout=dropout).to(device)
        else:
            self.classifier =  classifiers[cls_type](feat_dim,num_layer,fc,n_patches=None,dropout=dropout).to(device)
    def forward(self, x):
        x = x.view(x.size(0), self.channels, self.img_size, self.img_size)
        # print(f"Image10Classifier: x.shape={x.shape}")
        x = self.fft.forward(x)
        b=x.size(0)
        x = x.view(b, -1)
        if self.enc_type != 'none':
            # x = self.bn(x)
            x = self.ln(x)
            x = self.encoder(x.view(b, -1)) 
        x = self.classifier(x,b)
        return x

#--------------------------------------------------------------------

class Table10Classifier(nn.Module):#10クラスの表データ用
    def __init__(self, dataset,kernel_size,leverage,
                 enc_type,alpha,cls_type,num_layer,fc,dropout,device):
        super(Table10Classifier, self).__init__()
        dataset_config = {
            'covtype' : {'input_dim': 54}
        }
        encoders = {
            'PM':PMEncoder,
            'IM':IMEncoder,
            'MZM':MZMEncoder,
            'LI':LIEncoder,
            'none':PMEncoder
        }
        classifiers = {
            'MLP':MLP_for_7,
            'CNN':CNN_for10
        }
        self.input_dim = dataset_config[dataset]['input_dim']
        potential_dim = int(self.input_dim//leverage)
        self.enc_type = enc_type
        self.encoder = encoders[enc_type](self.input_dim,potential_dim,alpha,device)
        self.classifier =  classifiers[cls_type](potential_dim,num_layer,fc,n_patches=None,dropout=dropout).to(device)

    def forward(self, x):
        b=x.size(0)
        if self.enc_type != 'none':
            x = self.encoder(x)
        x = self.classifier(x,b)
        return x 


class DEQ_Image10Classifier(nn.Module):#10クラスの画像用(DEQ)
    def __init__(self, dataset,kernel_size,leverage,
                 enc_type,alpha,cls_type,num_layer,fc,dropout,num_iter,m,tol,beta,gamma,lam,device):
        super(DEQ_Image10Classifier, self).__init__() #DEQ_Image10Classifier, self
        self.device = device
        dataset_config = {
            'mnist':     {'img_size': 28, 'channels': 1},
            'cifar-10':  {'img_size': 32, 'channels': 3},
            'cinic-10': {'img_size':32, 'channels':3},
            'fashion-mnist': {'img_size': 28, 'channels': 1},
            'cifar-100': {'img_size': 32, 'channels': 3},
        }
        self.img_size = dataset_config[dataset]['img_size']
        self.channels = dataset_config[dataset]['channels']

        self.kernel_size = kernel_size
        kernel_in = self.channels*kernel_size**2
        classifiers = {
            'MLP':MLP_for_10,
            'CNN':CNN_for10
        }
        self.num_patches = (self.img_size//kernel_size)*(self.img_size//kernel_size) 
        kernel_in_total = kernel_in * self.num_patches
        # self.z_dim = int(kernel_in/leverage)(anderson軽量化)
        self.z_dim = int(kernel_in_total/leverage)
        # potential_dim = self.num_patches * self.z_dim(anderson軽量化)
        potential_dim = self.z_dim
        self.num_iter = num_iter
        #--------------------------------------------
        cell = Cell(kernel_in_total, self.z_dim,enc_type,alpha,gamma,device).to(device)
        self.deq_main = DEQFixedPoint(cell,anderson,self.z_dim,
                                      m = m,
                                      num_iter = num_iter,
                                      tol = tol,
                                      beta = beta,
                                      lam = lam
                                      )
        # print(f"DEQ_Image10Classifier: z_dim={self.z_dim}, num_patches={self.num_patches}, potential_dim={potential_dim}")

        self.classifier =  classifiers[cls_type](potential_dim,num_layer,fc,self.num_patches,dropout).to(device)
        
    def forward(self, x):
        b=x.size(0)
        x = x.view(b, self.channels, self.img_size, self.img_size)
        x = split_into_kernels(x, self.kernel_size)#(b, p, c, k, k)

        x = x.reshape(b,-1)

        x = self.deq_main(x)
        x = x.reshape(b * self.num_patches,-1)
        x = self.classifier(x,b)
        return x

class DEQ_Image10Classifier_FFT(nn.Module):#10クラスの画像用(DEQ)
    def __init__(self, dataset,kernel_size,leverage,
                 enc_type,alpha,cls_type,num_layer,fc,dropout,num_iter,m,tol,beta,gamma,lam,device):
        super(DEQ_Image10Classifier_FFT, self).__init__() 
        self.device = device
        dataset_config = {
            'mnist':     {'img_size': 28, 'channels': 1},
            'cifar-10':  {'img_size': 32, 'channels': 3},
            'cinic-10': {'img_size':32, 'channels':3},
            'fashion-mnist': {'img_size': 28, 'channels': 1},
            'cifar-100': {'img_size': 32, 'channels': 3},
        }
        self.img_size = dataset_config[dataset]['img_size']
        self.channels = dataset_config[dataset]['channels']
        classifiers = {
            'MLP':MLP_for_10,
            'CNN':CNN_for10
        }
        self.fft_dim = 25 #fft特徴量の次元数
        feat_dim = 17
        circuit_dim = 7 #積和演算回路の出力次元数
        self.num_iter = num_iter
        self.fft = FFTLowFreqSelector(out_dim=self.fft_dim, log_magnitude=True)
        self.bn = nn.BatchNorm1d(self.fft_dim)
        self.ln = nn.LayerNorm(self.fft_dim)
        #--------------------------------------------
        cell = Cell_fft(x_dim=self.fft_dim,circuit_dim=circuit_dim, z_dim=feat_dim,enc_type=enc_type,alpha=alpha,device=device).to(device)
        self.deq_main = DEQFixedPoint(cell,anderson,
                                      z_dim=feat_dim,
                                      m = m,
                                      num_iter = num_iter,
                                      tol = tol,
                                      beta = beta,
                                      lam = lam
                                      )

        self.classifier =  classifiers[cls_type](feat_dim,num_layer,fc,n_patches=None,dropout=dropout).to(device)
        
    def forward(self, x):
        x = x.view(x.size(0), self.channels, self.img_size, self.img_size)
        x = self.fft.forward(x)
        b = x.size(0)
        x = x.view(b,-1)
        # x = self.bn(x)
        x = self.ln(x)
        x = self.deq_main(x)
        x = self.classifier(x,b)
        return x


class DEQ_Table10Classifier(nn.Module):
    def __init__(self, dataset,kernel_size,leverage,
                 enc_type,alpha,cls_type,num_layer,fc,dropout,num_iter,m,tol,beta,gamma,lam,device):
        super(DEQ_Table10Classifier, self).__init__()
        self.device = device
        dataset_config = {
            'covtype': {'input_dim': 54}
        }
        classifiers = {
            'MLP':MLP_for_7
        }
        self.input_dim = dataset_config[dataset]['input_dim']
        self.z_dim = int(self.input_dim/leverage)
        potential_dim = self.z_dim
        self.num_iter = num_iter
        #--------------------------------------------
        cell = Cell(self.input_dim, self.z_dim,enc_type,alpha,gamma,device).to(device)
        self.deq_main = DEQFixedPoint(cell,anderson,self.z_dim,
                                      m = m,
                                      num_iter = num_iter,
                                      tol = tol,
                                      beta = beta,
                                      lam = lam
                                      )

        self.classifier =  classifiers[cls_type](potential_dim,num_layer,fc,n_patches=None,dropout=dropout).to(device)
        
    def forward(self, x):
        b=x.size(0)
        x = self.deq_main(x)
        x = self.classifier(x,b)
        return x