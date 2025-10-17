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
def normalize_zero_one(x, eps=1e-8):
    # x を [0,1] に正規化
    eps = 1e-8
    xmin = x.min(dim=1, keepdim=True)[0]
    xmax = x.max(dim=1, keepdim=True)[0]
    x = (x - xmin) / (xmax - xmin + eps)
    return x

#--------------------------------------------------------------------
def _rand_unitary(n, device=None, dtype=torch.cfloat):
    """複素ガウス→QRでHaar近似のユニタリ行列を生成"""
    A = torch.randn(n, n, device=device) + 1j * torch.randn(n, n, device=device) 
    Q, R = torch.linalg.qr(A)
    # 対角成分の位相で正規化
    d = torch.diagonal(R)
    ph = d / torch.abs(d)
    Q = Q @ torch.diag(ph.conj())
    return Q.to(dtype)

class PMEncoder(nn.Module):
    """
    Photonic phase-mod encoder: [B, input_dim] -> [B, output_dim]
    - 位相: φ = α ⊙ x （xは[0,1]想定）
    - 入力場: E_in = A * exp(i φ)  （Aは等振幅）
    - チップ: ユニタリUの上位 output_dim 行を観測行列Bとして使用
    - 出力: I = |E_in @ B^T|^2  （PD強度） 
    """
    def __init__(self, input_dim, output_dim, alpha=2*math.pi,device='cpu', seed=None): 
        super().__init__()
        device = torch.device(device)
        self.input_dim = input_dim
        self.output_dim = output_dim

        if seed is not None:
            torch.manual_seed(seed)

        # ランダムユニタリの行を切り出して観測行列Bに（受動干渉の部分観測モデル）
        U = _rand_unitary(input_dim, device=device)         # [in, in]
        B = U[:output_dim, :]                               # [out, in]
        self.register_buffer("B", B)                        # 固定

        # 位相係数 α と バイアス β（チャネル別）
        alpha_t = torch.full((1, input_dim), float(alpha), dtype=torch.float32, device=device)
        # alpha_t = (torch.rand(input_dim) - 0.5) * (2*alpha) 
        # self.register_buffer("alpha", alpha_t)
        # alpha の ±10% の範囲でランダムな値を生成
        # alpha_t = torch.rand(1, input_dim, dtype=torch.float32, device=device) * (0.2 * float(alpha)) + (0.9 * float(alpha)) 
        self.register_buffer("alpha", alpha_t.to(device))  # 固定
        # 入力振幅（総パワー一定なら 1/√N が無難）
        amp = torch.full((1, input_dim), 1.0 / math.sqrt(input_dim), dtype=torch.float32, device=device)
        self.register_buffer("amp", amp)

    def forward(self, x):
        # x: [B, input_dim]（0..1の実数）
        x = x.to(self.alpha.device, dtype=torch.float32)

        # 位相 φ = α⊙x
        phi = x * self.alpha                        # [B, in], 実数
        # 入力場 E_in = A * exp(i φ)
        E_in = self.amp * torch.exp(1j * phi)       # [B, in], 複素
        # 出力場
        E_out = E_in @ self.B.transpose(0, 1)       # [B, out], 複素
        # PD強度
        I = (E_out.abs() ** 2)                      # [B, out], 実・非負
        return I

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
                 enc_type,alpha,cls_type,num_layer,fc,ex_type,dropout,fft_params,device):
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
        # print("Image10Classifier: x.shape=",x.shape)
        b = x.size(0)
        x = x.view(b, self.channels, self.img_size, self.img_size)
        # print(f"Image10Classifier: x reshaped to (b,c,h,w): x.shape={x.shape}")
        x = self.split(x, self.kernel_size)#(b, p, c, k, k)
        # print(f"Image10Classifier: x split into patches: x.shape={x.shape}")
        x = x.reshape(b * self.num_patches,
                    self.channels * self.kernel_size**2)
        if self.enc_type != 'none': 
            x = normalize_zero_one(x) 
            # print("Before Encoder: x.shape=",x.shape)
            x = self.encoder(x) 
            # print("After Encoder: x.shape=",x.shape)
        x = self.classifier(x,b) 
        return x

#--------------------------------------------------------------------
class Image10Classifier_FFT(nn.Module):#10クラスの画像用(FFT特徴量版)
    def __init__(self, dataset,kernel_size,leverage,
                 enc_type,alpha,cls_type,num_layer,fc,ex_type,dropout,fft_params,device):
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
        self.device = device
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
        self.fft_dim = fft_params['fft_dim'] #fft特徴量の次元数
        feat_dim = fft_params['enc_out'] #encoder出力の次元数
        self.compressed_dim = fft_params['compressed_dim'] #圧縮後の次元数
        self.enc_type = enc_type
        self.ex_type = ex_type
        self.fft = FFTLowFreqSelector(out_dim=self.fft_dim, log_magnitude=True)
        self.bn = nn.BatchNorm1d(self.fft_dim).to(device)
        self.ln = nn.LayerNorm(self.fft_dim, elementwise_affine=False).to(device)
        self.encoder = encoders[enc_type](self.fft_dim,feat_dim,alpha,device=device) 

        self._selected_cols = None  # ランダムに選んだ列のインデックスを保存するための変数

        self.classifier =  classifiers[cls_type](self.compressed_dim,num_layer,fc,n_patches=None,dropout=dropout).to(device)

    def random_subarray(self, arr, m: int):
        """
        2D tensor/ndarray arr (batch, K) から列をランダムに m 個選んで (batch, m) を返す（torch.Tensor を返す）。
        - 入力が numpy.ndarray の場合は torch.Tensor に変換する。
        """
        # numpy を受け取ったら tensor に変換
        if isinstance(arr, np.ndarray):
            arr = torch.from_numpy(arr)
        if not torch.is_tensor(arr):
            arr = torch.tensor(arr)
        if arr.dim() != 2:
            raise ValueError(f"arr must be 2D (batch, features), got dim={arr.dim()}")
        batch, K = arr.shape
        if m > K:
            raise ValueError(f"m ({m}) must be smaller than number of features ({K})")
        if m == K:
            return arr  # 全列選択ならそのまま返す
        if self._selected_cols is None or self._selected_cols.numel() != m or self._selected_cols.max().item() >= K:
            cols = torch.randperm(K, device=self.device)[:m]
            cols, _ = torch.sort(cols)  # 元の列順を保つ
            self._selected_cols = cols.to(self.device).clone().detach()
        return arr[:, self._selected_cols]
    
    def forward(self, x):
        # print("Image10Classifier_FFT: x.shape=",x.shape)
        if self.ex_type == 'fft':
            x = x.view(x.size(0), self.channels, self.img_size, self.img_size)
            # print(f"Image10Classifier: x.shape={x.shape}")
            x = self.fft.forward(x)
        b=x.size(0)
        x = x.view(b, -1)
        if self.enc_type != 'none':
            # x = self.bn(x)
            # x = self.ln(x)
            #--------------------------------------------
            eps = 1e-8
            xmin = x.min(dim=1, keepdim=True)[0]
            xmax = x.max(dim=1, keepdim=True)[0]
            x = (x - xmin) / (xmax - xmin + eps)
            #--------------------------------------------
            # print("x after normalization:", x)
            x = self.encoder(x.view(b, -1)) 
        # print("After Encoder: x.shape=",x.shape)
        x = self.random_subarray(x, self.compressed_dim)
        # print("After Subarray: x.shape=",x.shape)
        x = self.classifier(x,b)
        return x

class Table10Classifier(nn.Module):#10クラスの表データ用
    def __init__(self, dataset,kernel_size,leverage,
                 enc_type,alpha,cls_type,num_layer,fc,ex_type,dropout,fft_params,device):
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
            x = normalize_zero_one(x)
            x = self.encoder(x)
        x = self.classifier(x,b)
        return x 

class DEQ_Image10Classifier(nn.Module):#10クラスの画像用(DEQ)
    def __init__(self, dataset,kernel_size,leverage,
                 enc_type,alpha,cls_type,num_layer,fc,ex_type,dropout,num_iter,m,tol,beta,gamma,lam,device):
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
                 enc_type,alpha,cls_type,num_layer,fc,ex_type,dropout,num_iter,m,tol,beta,gamma,lam,device):
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
        self.ln = nn.LayerNorm(self.fft_dim, elementwise_affine=False)
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
                 enc_type,alpha,cls_type,num_layer,fc,ex_type,dropout,num_iter,m,tol,beta,gamma,lam,device):
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