import torch
import torch.nn as nn

from .IntegrationModel import PMEncoder, IMEncoder, MZMEncoder, LIEncoder 

class Cell(nn.Module):
    def __init__(self, x_dim, z_dim,enc_type,alpha,device):
        super().__init__()
        encoders = {
            'PM':PMEncoder,
            'IM':IMEncoder,
            'MZM':MZMEncoder,
            'LI':LIEncoder
        }
        self.enc1 = encoders[enc_type](x_dim+z_dim,z_dim,alpha,device)
        self.fc1 = nn.Linear(z_dim,z_dim)
        self.bn = nn.BatchNorm1d(z_dim)
        self.act = nn.ReLU()
    def forward(self,z , x):
        zx = torch.cat([x,z],dim=1)
        z = self.enc1(zx)
        #以下、積和演算電子回路-----------------------
        z = self.bn(z)
        z = self.fc1(z)
        z = self.act(z)
        return z

def anderson(fc, x0,z_dim, m, num_iter, tol, beta, lam=1e-4):
    bsz,_ = x0.shape
    X = torch.zeros(bsz, m, z_dim, dtype=x0.dtype, device=x0.device)
    F = torch.zeros_like(X)
    X[:, 0], F[:, 0] = x0, fc(x0)
    X[:, 1], F[:, 1] = F[:, 0], fc(F[:, 0])

    H_mat = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H_mat[:, 0, 1:] = H_mat[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    res = []  # 収束誤差の履歴
    for k in range(2, num_iter):
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H_mat[:, 1:n + 1, 1:n + 1] = torch.bmm(G, G.transpose(1, 2)) + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[None]
        try:
            alpha = torch.linalg.lstsq( H_mat[:, :n+1, :n+1], y[:, :n+1]).solution[:, 1:n+1, 0] #最小2乗解
            # alpha = torch.linalg.solve(H_mat[:, :n + 1, :n + 1], y[:, :n + 1])[:, 1:n + 1, 0]
        except RuntimeError as e:
            print(f"[Warn] Skipping batch {bsz} at iter {k} due to singular matrix.")
            continue 
        X[:, k % m] = beta * (alpha[:, None] @ F[:, :n]).squeeze(1) + (1 - beta) * (alpha[:, None] @ X[:, :n]).squeeze(1)
        x_current = X[:,k%m]
        F[:, k % m] = fc(X[:,k%m])
        # 残差のノルムを計算（収束判定用）
        res_norm = (F[:, k % m] - X[:, k % m]).norm().item() / (1e-5 + F[:, k % m].norm().item())
        res.append(res_norm)
        if res[-1] < tol:
            break
    return X[:, k % m], res

class DEQFixedPoint(nn.Module):
    def __init__(self, f, solver, z_dim,**kwargs):
        """
        f: 固定点関数（FixedPointLayer）
        solver: 固定点反復のソルバー（anderson関数）
        kwargs: ソルバーに渡す追加パラメータ
        """
        super().__init__()
        self.f = f
        self.solver = solver
        self.z_dim =z_dim
        self.kwargs = kwargs

    def forward(self, x):
        bsz = x.size(0)
        z0 = torch.zeros(bsz, self.z_dim, dtype=x.dtype, device=x.device)
        with torch.no_grad():
            z, self.forward_res = self.solver(lambda z: self.f(z,x), z0
                                              ,z_dim=self.z_dim, **self.kwargs)
        # 得られた z を再度 f に通し、最終出力を計算
        z = self.f(z, x)
        z.requires_grad_()
        # 逆伝播用に z のコピーを作成し、再度 f(z, x) を計算
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0, x)
        
        # 逆伝播時のフックを定義
        def backward_hook(grad):
            g_st, _ = self.solver(lambda y: torch.autograd.grad(f0, z0, y, retain_graph=True)[0] + grad, grad,z_dim=self.z_dim, **self.kwargs)
            return g_st

        z.register_hook(backward_hook)
        return z
