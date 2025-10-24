import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import matplotlib.pyplot as plt

class SNLinearRelax(nn.Linear):
    def __init__(
        self, in_features, out_features, bias=True,
        gamma=0.9, n_power_iterations=5):
        super().__init__(in_features, out_features, bias)
        # 最大特異値を ≈1 へ
        spectral_norm(self, name='weight', n_power_iterations=n_power_iterations)
        # γ を登録（勾配は流さないので buffer で十分）
        self.register_buffer("gamma", torch.tensor(float(gamma)))

    def forward(self, x):
        # 本来の weight はすでに spectral_norm で正規化済み
        return F.linear(x, self.gamma * self.weight, self.bias)

class Cell(nn.Module):
    def __init__(self, x_dim, z_dim,enc_type,alpha,gamma,device):
        super().__init__()
        # 循環インポートを避けるため、ここでインポート
        from .IntegrationModel import PMEncoder, IMEncoder, MZMEncoder, LIEncoder
        
        encoders = {
            'PM':PMEncoder,
            'IM':IMEncoder,
            'MZM':MZMEncoder,
            'LI':LIEncoder
        }
        self.enc1 = encoders[enc_type](x_dim+z_dim,z_dim,alpha,device)
        # self.fc1 = spectral_norm(nn.Linear(z_dim, z_dim))
        self.fc1 = nn.Linear(z_dim, z_dim)
        # self.fc1 = SNLinearRelax(z_dim, z_dim, gamma=gamma)

        # self.bn = nn.BatchNorm1d(z_dim)
        self.ln = nn.LayerNorm(z_dim)#,elementwise_affine=False)
        self.act = nn.ReLU()
        # self.act = nn.Tanh()
    def forward(self,z , x):
        zx = torch.cat([x,z],dim=1)
        # print(f"Cellzx:{zx.shape}")
        z = self.enc1(zx)
        #以下、積和演算電子回路-----------------------
        z = self.ln(z)
        z = self.fc1(z)
        # print(f"Cellz:{z.shape}")
        return z


class Cell_fft(nn.Module):
    def __init__(self, x_dim,circuit_dim, z_dim,enc_type,alpha,device):
        super().__init__()
        from .IntegrationModel import PMEncoder, IMEncoder, MZMEncoder, LIEncoder
        
        encoders = {
            'PM':PMEncoder,
            'IM':IMEncoder,
            'MZM':MZMEncoder,
            'LI':LIEncoder
        }
        self.enc1 = encoders[enc_type](x_dim+circuit_dim,z_dim,alpha,device)
        self.fc1 = nn.Linear(z_dim, circuit_dim)
        self.bn = nn.BatchNorm1d(z_dim)
        self.act = nn.ReLU()
    def forward(self,z , x):
        # print(f"Cell_fft: x.shape={x.shape}, z.shape={z.shape}")
        #積和演算電子回路----------------------
        if z.shape[0] != 1:  # バッチサイズが1でない場合のみBatchNormを適用
            z = self.bn(z)
        z = self.fc1(z) #17→7
        z = self.act(z)
        #------------------------------------
        zx = torch.cat([x,z],dim=1)
        z = self.enc1(zx)#光回路:32→17で固定
        return z
        

def anderson(fc, x0, z_dim, m, num_iter, tol, beta, lam=1e-4):
    bsz,_ = x0.shape
    dtype, device = x0.dtype, x0.device
    X = torch.zeros(bsz, m, z_dim, dtype=dtype, device=device)
    F = torch.zeros_like(X)
    X[:, 0], F[:, 0] = x0, fc(x0)
    
    # num_iter=1の場合は初期値をそのまま返す
    if num_iter <= 1:
        return X[:, 0], []
        
    X[:, 1], F[:, 1] = F[:, 0], fc(F[:, 0])
    
    # num_iter=2の場合は2回目の結果を返す
    if num_iter <= 2:
        return X[:, 1], []

    H_mat = torch.zeros(bsz, m + 1, m + 1, dtype=dtype, device=device)
    H_mat[:, 0, 1:] = H_mat[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=dtype, device=device)
    y[:, 0] = 1

    res = []  # 収束誤差の履歴
    for k in range(2, num_iter):
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H_mat[:, 1:n + 1, 1:n + 1] = torch.bmm(G, G.transpose(1, 2)) + lam * torch.eye(n, dtype=dtype, device=device)[None]
        try:
            alpha = torch.linalg.lstsq( H_mat[:, :n+1, :n+1], y[:, :n+1]).solution[:, 1:n+1, 0] #最小2乗解
            # alpha = torch.linalg.solve(H_mat[:, :n + 1, :n + 1], y[:, :n + 1])[:, 1:n + 1, 0]
        except RuntimeError as e:
            print(f"[Warn] Skipping batch {bsz} at iter {k} due to singular matrix.")
            continue 
        X[:, k % m] = beta * (alpha[:, None] @ F[:, :n]).squeeze(1) + (1 - beta) * (alpha[:, None] @ X[:, :n]).squeeze(1)
        F[:, k % m] = fc(X[:, k % m])
        # 残差のノルムを計算（収束判定用）
        res_norm = (F[:, k % m] - X[:, k % m]).norm().item() / (1e-5 + F[:, k % m].norm().item())
        res.append(res_norm)
        if res[-1] < tol:
            break
    else:
        k = num_iter - 1  # 最大反復に達した場合
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

#--------------------------------------------------------
class FFTLowFreqSelector(torch.nn.Module):
    """
    MNIST/FashionMNIST 等の2D画像を FFT し、fftshift 後の中心(低周波)から
    半径の近い順に out_dim 個の周波数成分 (振幅) を抜き出して特徴ベクトルにするクラス。

    - 入力: x (B, C, H, W)  [float]
    - 出力: (B, out_dim)    [float]
    """
    def __init__(self, out_dim: int = 25, log_magnitude: bool = True, eps: float = 1e-12):
        super().__init__()
        self.out_dim = int(out_dim)
        self.log_magnitude = bool(log_magnitude)
        self.eps = float(eps)
        # 画像サイズごとに選択インデックスをキャッシュ
        self._cached_idx = {}   # key=(H,W) -> dict{"flat": LongTensor(K,), "ij": LongTensor(K,2)}

    @torch.no_grad()
    def _prepare_indices(self, H: int, W: int, device: torch.device):
        """中心からの半径昇順（同半径は角度昇順）で K=out_dim 個の画素インデックスを決めてキャッシュ。"""
        if (H, W) in self._cached_idx:
            return
        # 座標グリッド（行=Y=i, 列=X=j）
        ys = torch.arange(H, device=device).float()
        xs = torch.arange(W, device=device).float()
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        cy = (H - 1) / 2.0
        cx = (W - 1) / 2.0
        dy = yy - cy
        dx = xx - cx
        r2 = dy * dy + dx * dx
        ang = torch.atan2(dy, dx)  # -pi..pi

        # (H*W,) にして複合キーでソート
        r2_f = r2.flatten()
        ang_f = ang.flatten()
        # 距離→角度のタプルで安定ソートするため、まず距離で昇順、距離同値は角度で昇順
        # 距離を主キー、角度を副キーにするために、距離に極小の角度正規化を足す手もあるが、
        # ここでは2段階ソートを使う。
        # 1) 角度でソートインデックス
        _, idx_ang = torch.sort(ang_f)
        # 2) 上の結果を使って距離で安定ソート
        r2_sorted, idx_r2 = torch.sort(r2_f[idx_ang], stable=True)
        idx_all = idx_ang[idx_r2]

        K = min(self.out_dim, H * W)
        idx_k = idx_all[:K]  # (K,)

        # 2次元インデックス (i,j)
        iy = (idx_k // W).long()
        ix = (idx_k %  W).long()
        ij = torch.stack([iy, ix], dim=1).long()

        self._cached_idx[(H, W)] = {"flat": idx_k.long(), "ij": ij.long()}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) または (B, H, W)/(B, H*W)
        return: (B, out_dim)
        """
        B, C, H, W = x.shape
        if self.out_dim > H * W:
            raise ValueError(f"out_dim={self.out_dim} exceeds image size H*W={H*W}")

        self._prepare_indices(H, W, x.device)
        idx_flat = self._cached_idx[(H, W)]["flat"]  # (K,)
        # FFT -> shift -> magnitude
        X = torch.fft.fft2(x, dim=(-2, -1))
        X = torch.fft.fftshift(X, dim=(-2, -1))
        mag = X.abs()
        if self.log_magnitude:
            mag = torch.log1p(mag + self.eps)

        # (B, C, H*W) にして、同一 idx_flat を切り出し (B, C, K) -> チャンネル平均 -> (B, K)
        mag_flat = mag.reshape(B, C, H * W)
        feats_bcK = torch.index_select(mag_flat, dim=2, index=idx_flat)  # (B, C, K)
        feats = feats_bcK.mean(dim=1)  # (B, K)
        return feats

    @torch.no_grad()
    def get_selected_coords(self, H: int, W: int, device=None) -> torch.Tensor:
        """選ばれる (i,j) インデックス（K,2）を返す。"""
        dev = device if device is not None else torch.device("cpu")
        self._prepare_indices(H, W, dev)
        return self._cached_idx[(H, W)]["ij"].clone()

    @torch.no_grad()
    def reconstruct_from_lowfreq(self, x: torch.Tensor, sample_index: int = 0) -> torch.Tensor:
        """
        選択した低周波成分のみを使用して逆フーリエ変換で画像を復元する。
        
        Args:
            x: 入力画像テンソル (B, C, H, W)
            sample_index: 復元するサンプルのインデックス
            
        Returns:
            復元された画像 (H, W)
        """
        B, C, H, W = x.shape
        if not (0 <= sample_index < B):
            raise IndexError(f"sample_index {sample_index} out of range (0..{B-1})")
            
        self._prepare_indices(H, W, x.device)
        idx_flat = self._cached_idx[(H, W)]["flat"]  # (K,)
        ij = self._cached_idx[(H, W)]["ij"]  # (K, 2)
        
        # 1サンプルのFFT
        x_single = x[sample_index:sample_index+1]  # (1, C, H, W)
        X = torch.fft.fft2(x_single, dim=(-2, -1))
        X_shifted = torch.fft.fftshift(X, dim=(-2, -1))
        
        # 低周波成分のみを保持するマスク作成
        X_lowfreq = torch.zeros_like(X_shifted)
        
        # 選択された低周波成分のみをコピー
        for c in range(C):
            for k in range(len(idx_flat)):
                i, j = ij[k]
                X_lowfreq[0, c, i, j] = X_shifted[0, c, i, j]
        
        # 逆フーリエ変換
        X_lowfreq_ishifted = torch.fft.ifftshift(X_lowfreq, dim=(-2, -1))
        reconstructed = torch.fft.ifft2(X_lowfreq_ishifted, dim=(-2, -1)).real
        
        # チャンネル平均して (H, W) にする
        reconstructed = reconstructed.mean(dim=1).squeeze(0)  # (H, W)
        
        return reconstructed

    @torch.no_grad()
    def plot_example(self, x: torch.Tensor, sample_index: int = 0, annotate: bool = True, savepath: str = None):
        """
        バッチ x から sample_index 番目を可視化。
        ・元画像
        ・FFT(shift後)の log 振幅スペクトル
        ・どの out_dim 成分を抜いたか（スペクトル上にマーカー）
        ・低周波成分からの復元画像
        """
        B, C, H, W = x.shape
        if not (0 <= sample_index < B):
            raise IndexError(f"sample_index {sample_index} out of range (0..{B-1})")
        
        # 1枚取り出し（表示は1ch目）
        img_tensor = x[sample_index, 0].detach().cpu().float()  # (H, W)
        # 正規化を戻す: Normalize((0.5,), (0.5,)) の逆変換
        img = img_tensor * 0.5 + 0.5  # [-1,1] -> [0,1] に戻す
        img = torch.clamp(img, 0, 1)  # 範囲をクリップ
        img = img.numpy()

        # FFT スペクトル（log1p 振幅）
        X = torch.fft.fft2(x[sample_index:sample_index+1], dim=(-2, -1))
        X = torch.fft.fftshift(X, dim=(-2, -1))
        mag = X.abs().mean(dim=1)  # (1, H, W) チャンネル平均
        spec = torch.log1p(mag + self.eps)[0].detach().cpu().float().numpy()

        # 低周波成分からの復元画像
        reconstructed = self.reconstruct_from_lowfreq(x, sample_index)
        # 正規化を戻す
        reconstructed_denorm = reconstructed * 0.5 + 0.5
        reconstructed_denorm = torch.clamp(reconstructed_denorm, 0, 1)
        reconstructed_img = reconstructed_denorm.detach().cpu().numpy()

        # マーカー座標
        ij = self.get_selected_coords(H, W, device=x.device).cpu().numpy()  # (K,2)
        iy, ix = ij[:, 0], ij[:, 1]

        # 図を描く（4つのサブプロット）
        fig = plt.figure(figsize=(16, 4))
        ax1 = fig.add_subplot(1, 4, 1)
        ax2 = fig.add_subplot(1, 4, 2)
        ax3 = fig.add_subplot(1, 4, 3)
        ax4 = fig.add_subplot(1, 4, 4)

        ax1.imshow(img, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
        ax1.set_title("Original")
        ax1.axis("off")

        ax2.imshow(spec, cmap="gray", interpolation="nearest")
        ax2.set_title("FFT |X| (log)")
        ax2.axis("off")

        ax3.imshow(spec, cmap="gray", interpolation="nearest")
        ax3.scatter(ix, iy, marker="o", s=30, c='red', alpha=0.7)
        if annotate:
            for k, (yy, xx) in enumerate(zip(iy, ix)):
                ax3.text(xx + 0.5, yy + 0.5, str(k+1), fontsize=7, color='blue')
        ax3.set_title(f"Selected {self.out_dim} low-freq bins")
        ax3.axis("off")

        ax4.imshow(reconstructed_img, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
        ax4.set_title("Reconstructed from low-freq")
        ax4.axis("off")

        plt.tight_layout()
        if savepath:
            plt.savefig(savepath, bbox_inches="tight", dpi=150)
        plt.show()