import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager
import platform

# フォント設定を改善（macOSでの日本語対応）
_available_fonts = {f.name for f in font_manager.fontManager.ttflist}
system_type = platform.system()

if system_type == "Darwin":  # macOS
    # macOSで利用可能な日本語フォントを優先順位で設定
    japanese_fonts = [
        'Hiragino Sans',
        'Hiragino Kaku Gothic ProN',
        'Hiragino Mincho ProN',
        'Osaka',
        'AppleGothic',
        'Noto Sans CJK JP',
        'Yu Gothic',
        'DejaVu Sans'
    ]
    
    # 利用可能なフォントから最初に見つかったものを選択
    selected_font = 'DejaVu Sans'  # デフォルト
    for font in japanese_fonts:
        if font in _available_fonts:
            selected_font = font
            break
    
    rcParams['font.family'] = selected_font
    rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け防止
else:
    # その他のOS（従来の設定）
    rcParams['font.family'] = 'Times New Roman' if 'Times New Roman' in _available_fonts else 'DejaVu Serif'

import numpy as np
import pandas as pd
import seaborn as sns
import csv
from sklearn.metrics import confusion_matrix
import random
import tempfile
from dataloader.dataloader import get_new_dataloader 
from models.IntegrationModel import split_into_kernels, PMEncoder,IMEncoder,MZMEncoder,LIEncoder,DEQ_Image10Classifier
from models.OtherModels import Cell,Cell_fft,anderson,FFTLowFreqSelector
colors = [
    "#1f77b4",  # 青
    "#ff7f0e",  # オレンジ
    "#2ca02c",  # 緑
    "#d62728",  # 赤
    "#9467bd",  # 紫
    "#8c564b",  # 茶
    "#e377c2",  # ピンク
    "#7f7f7f",  # グレー
    "#bcbd22",  # 黄緑
    "#17becf",  # 水色
]


#------------------------------------------------------------------------------------------
def plot_loss_curve(train_loss,test_loss,Save=False,Show=False):
    plt.figure(figsize=(6,4.5))
    plt.plot(range(1,len(train_loss)+1),train_loss,label='training',color='blue')
    plt.plot(range(1,len(test_loss)+1),test_loss,label='test',color='cyan')
    plt.xlabel('Epochs')
    plt.ylabel('LOSS')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    if Show:
        plt.show()
    if Save:
        tmp_img = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        plt.savefig(tmp_img.name)
        plt.close(plt.gcf())
        return tmp_img.name
        
#------------------------------------------------------------------------------------------
def plot_errorbar_losscurve(All_loss_test, Save=False, Show=False):
    epochs = len(All_loss_test[0]) 
    num_dimensions = len(All_loss_test)
    mean_loss = np.mean(All_loss_test, axis=0)
    std_loss = np.std(All_loss_test, axis=0)

    fig, ax1 = plt.subplots()

    ax1.errorbar(
        x=range(1, epochs + 1), y=mean_loss, yerr=std_loss,
        fmt='-o', color='blue', ecolor='blue', capsize=5, 
    )

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('LOSS', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    plt.title('LOSS Transition in Test data')
    if Show:
        plt.show()
    if Save:
        tmp_img = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        plt.savefig(tmp_img.name)
        plt.close(plt.gcf())
        return tmp_img.name

#------------------------------------------------------------------------------------------
def plot_confusion_matrix(true_labels,pred_labels,dataset,test_acc,Save=False,Show=False):

    num_labels = {
        'mnist':range(10),
        'cifar-10': ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
        'cinic-10': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
        'fashion-mnist':['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
        'covtype':["Spruce/Fir","Lodgepole Pine","Ponderosa Pine",
                   "Cottonwood/Willow","Aspen","Douglas-fir","Krummholz" ]
    }
    cm = confusion_matrix(true_labels,pred_labels)
    cm = cm.astype('float')/cm.sum(axis=1,keepdims = True)
    plt.figure(figsize=(6,4.5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=num_labels[dataset],yticklabels=num_labels[dataset])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f"Overall Correction Rate:{test_acc:.2f}%")
    if Show:
        plt.show()
    if Save:
        tmp_img = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        plt.savefig(tmp_img.name)
        plt.close(plt.gcf())
        return tmp_img.name
#------------------------------------------------------------------------------------------
def plot_histograms(data_train,data_test, dataset, kernel_size, batch_size,enc_type): #画像専用
    encoders = {
        'PM':PMEncoder,
        'IM':IMEncoder,
        'MZM':MZMEncoder,
        'LI':LIEncoder
    }
    dataset_config = {
        'mnist':     {'img_size': 28, 'channels': 1},
        'cifar-10':  {'img_size': 32, 'channels': 3},
        'fashion-mnist': {'img_size': 28, 'channels': 1},
        'cifar-100': {'img_size': 32, 'channels': 3},
        'cinic-10': {'img_size': 32, 'channels': 3}
    }
    
    img_size = dataset_config[dataset]['img_size']
    channels = dataset_config[dataset]['channels']
    
    _, test_loader = get_new_dataloader(data_train,data_test,batch_size)
    x, _ = random.choice(test_loader.dataset)
    x = x.view(batch_size, channels, img_size, img_size)
    x_splitted = split_into_kernels(x, kernel_size)
    x_in_flat = x_splitted.reshape(-1).detach().cpu().numpy()
    x_encoded = encoders[enc_type](x_splitted)
    x_out_flat = x_encoded.reshape(-1).detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(x_in_flat, bins=20, color='darkorange', alpha=0.7)
    axes[0].set_xlabel("Input value", fontsize=14)
    axes[0].set_ylabel("Frequency", fontsize=14)
    axes[0].set_title("Input x Histogram", fontsize=14)
    axes[0].tick_params(labelsize=12)
    axes[0].set_ylim(0,)
    # エンコーダ出力ヒストグラム
    axes[1].hist(x_out_flat, bins=20, color='steelblue', alpha=0.7)
    axes[1].set_xlabel("Encoder Output value", fontsize=14)
    axes[1].set_ylabel("Frequency", fontsize=14)
    axes[1].set_title("Encoder Output Histogram", fontsize=14)
    axes[1].tick_params(labelsize=12)
    plt.tight_layout()
    plt.show()

#------------------------------------------------------------------------------------------
def create_table(All_test_acc,All_last_loss,All_pro_time,Save=False,Show=False):
    ACC_mean = np.mean(All_test_acc)
    ACC_best = np.max(All_test_acc)
    ACC_bestID = np.argmax(All_test_acc) + 1
    ACC_worst = np.min(All_test_acc)
    ACC_worstID = np.argmin(All_test_acc) + 1

    LOSS_mean = np.mean(All_last_loss)
    LOSS_best = np.min(All_last_loss)
    LOSS_bestID = np.argmin(All_last_loss) + 1
    LOSS_worst = np.max(All_last_loss)
    LOSS_worstID = np.argmax(All_last_loss) + 1

    PRO_mean = np.mean(All_pro_time)
    PRO_best = np.min(All_pro_time)  # 最短
    PRO_bestID = np.argmin(All_pro_time) + 1
    PRO_worst = np.max(All_pro_time) # 最長
    PRO_worstID = np.argmax(All_pro_time) + 1

    data = {
        "":       ["ACC",     "LOSS",     "TIME"],
        "Average":[f"{ACC_mean:.3f}", f"{LOSS_mean:.3f}", f"{PRO_mean:.3f}"],
        "Best ID":[ACC_bestID, LOSS_bestID, PRO_bestID],
        "Best":   [f"{ACC_best:.3f}", f"{LOSS_best:.3f}", f"{PRO_best:.3f}"],
        "Worst ID":[ACC_worstID, LOSS_worstID, PRO_worstID],
        "Worst":  [f"{ACC_worst:.3f}", f"{LOSS_worst:.3f}", f"{PRO_worst:.3f}"],
    }
    df = pd.DataFrame(data)
    if Show:
        print(df)
    if Save:
        return df
#------------------------------------------------------------------------------------------
def convergence_verify(params,gamma,data_train,data_test,device,Show=False):
    dataset = params['dataset']
    num_iter = params['num_iter']
    m = params['m']
    tol = params['tol']
    beta = params['beta']
    kernel_size = params['kernel_size']
    enc_type = params['enc_type']
    leverage = params['leverage']#[0]
    alpha = params['alpha']

    dataset_config = {
        'mnist':     {'img_size': 28, 'channels': 1},
        'cifar-10':  {'img_size': 32, 'channels': 3},
        'cinic-10': {'img_size':32, 'channels':3},
        'fashion-mnist': {'img_size': 28, 'channels': 1},
        'cifar-100': {'img_size': 32, 'channels': 3}
    }
    img_size = dataset_config[dataset]['img_size']
    channels = dataset_config[dataset]['channels']

    if kernel_size > 0:
        kernel_in = int(channels*kernel_size**2)

        # leverageがlistの場合は最初の要素を使用、intの場合はそのまま使用
        leverage_value = leverage[0] if isinstance(leverage, list) else leverage

        z_dim = int(kernel_in/leverage_value)
        num_patches = int(img_size/kernel_size)**2
        cell = Cell(kernel_in, z_dim,enc_type,alpha,gamma,device).to(device)
    else:
        leverage_value = leverage[0] if isinstance(leverage, list) else leverage
        in_dim = channels*img_size*img_size
        z_dim = int(in_dim / leverage_value)
        cell = Cell(in_dim, z_dim,enc_type,alpha,gamma,device).to(device)

    batch_size = 64
    _,test_dataloader = get_new_dataloader(data_train,data_test,batch_size)
    for x_batch, _ in test_dataloader:
        idx = random.randint(0, batch_size-1)
        x_sample = x_batch[idx].to(device)
        break
    # --------------------------------------------
    relres = []   # 相対残差の推移
    with torch.no_grad():
        B = 1
        x_sample = x_sample.view(B, channels,img_size, img_size)
        if kernel_size > 0:
            x_patch = split_into_kernels(x_sample, kernel_size)# (B, N, k, k)
            x_patch = x_patch.reshape(B * num_patches, -1)   # (B*N, in_dim)
            def fc(z):
                return cell(z,x_patch)
            z0 = torch.zeros(B * num_patches, z_dim, device=device)
        else:
            x_flat = x_sample.view(B, -1)  # (B, in_dim)
            def fc(z):
                return cell(z,x_flat)
            z0 = torch.zeros(B, z_dim, device=device)

        z_final, relres = anderson(
            fc,
            z0,
            z_dim=z_dim,
            m=m,
            num_iter=num_iter,
            tol=tol,
            beta=beta
        )
    # ---------------- プロット ----------------------------------
    if Show:
        plt.figure(figsize=(6,4))
        plt.semilogy(range(1, len(relres)+1), relres, marker="o")
        plt.xlabel("iteration")
        plt.ylabel("relative residual")
        plt.title("convergence (relative residual)")
        plt.grid(True, which="both")
        plt.tight_layout()
        plt.show()
    return relres

def convergence_verify_fft(params,gamma,data_train, data_test, device, Show=False):
    # パラメータ抽出
    dataset = params['dataset']
    num_iter = params['num_iter']
    m = params['m']
    tol = params['tol']
    beta = params['beta']
    enc_type = params['enc_type']
    alpha = params['alpha']
    
    dataset_config = {
        'mnist':     {'img_size': 28, 'channels': 1},
        'cifar-10':  {'img_size': 32, 'channels': 3},
        'cinic-10':  {'img_size': 32, 'channels': 3},
        'fashion-mnist': {'img_size': 28, 'channels': 1},
        'cifar-100': {'img_size': 32, 'channels': 3},
    }
    
    img_size = dataset_config[dataset]['img_size']
    channels = dataset_config[dataset]['channels']
    
    # FFTベースのパラメータ設定（IntegrationModel.pyのDEQ_Image10Classifier_FFTと同じ）
    fft_dim = 25        # FFT低周波成分数
    z_dim = 17          # 隠れ状態の次元
    circuit_dim = 7     # 積和演算電子回路の出力次元数
    
    # FFT特徴抽出器とCell_fftの初期化
    from models.OtherModels import Cell_fft, anderson, FFTLowFreqSelector
    fft_extractor = FFTLowFreqSelector(out_dim=fft_dim, log_magnitude=True)
    cell = Cell_fft(x_dim=fft_dim, circuit_dim=circuit_dim, z_dim=z_dim, 
                   enc_type=enc_type, alpha=alpha, device=device).to(device)
    
    # テストデータから1サンプル取得
    batch_size = 64
    _, test_dataloader = get_new_dataloader(data_train, data_test, batch_size)
    for x_batch, _ in test_dataloader:
        idx = random.randint(0, batch_size-1)
        x_sample = x_batch[idx].to(device)
        break
    
    # 収束検証の実行
    relres = []
    with torch.no_grad():
        B = 1
        # 画像形状に変換
        x_sample = x_sample.view(B, channels, img_size, img_size)
        
        # FFTで低周波成分を抽出
        x_fft_features = fft_extractor.forward(x_sample)  # (B, fft_dim)
        
        def fc(z):
            # z: (B, circuit_dim), x_fft_features: (B, fft_dim)
            return cell(z, x_fft_features)
        
        # 初期値設定
        z0 = torch.zeros(B, z_dim, device=device)
        
        # Anderson反復による固定点計算
        z_final, relres = anderson(
            fc,
            z0,
            z_dim=z_dim,
            m=m,
            num_iter=num_iter,
            tol=tol,
            beta=beta
        )
    
    # 結果の可視化
    if Show:
        plt.figure(figsize=(6, 4))
        plt.semilogy(range(1, len(relres)+1), relres, marker="o")
        plt.xlabel("iteration")
        plt.ylabel("relative residual")
        plt.title("FFT-based DEQ convergence (relative residual)")
        plt.grid(True, which="both")
        plt.tight_layout()
        plt.show()
    
    return relres


def convergence_verify_tabular(params, gamma, data_train, data_test, device, Show=False):
    # ---- params ----
    data_set    = params['dataset'] 
    num_iter    = params['num_iter']
    m           = params['m']
    tol         = params['tol']
    beta        = params['beta']
    enc_type    = params['enc_type']   # 'PM' / 'IM' / 'MZM' / 'LI' など
    leverage    = params['leverage']   # int か [int]
    alpha       = params['alpha']      # 位相感度など Cell に渡す

    # leverage が list なら先頭を採用
    leverage_value = leverage[0] if isinstance(leverage, list) else leverage

    data_size = {
        "covtype": 54
    }
    z_dim = int(data_size["covtype"] / leverage_value)
    cell = Cell(data_size["covtype"], z_dim, enc_type, alpha, gamma, device).to(device)

    # ---- dataloader から1バッチだけ取り、そこから1標本(or 小バッチ)を作る ----
    batch_size = 64
    _, test_loader = get_new_dataloader(data_train, data_test, batch_size)
    for x_batch, _ in test_loader:
        idx = random.randint(0, batch_size - 1)
        x_sample = x_batch[idx].to(device)  # 形状: (D,)
        break

    relres = []
    with torch.no_grad():
        B = 1
        x_sample = x_sample.view(B, data_size["covtype"])  # 形状: (B, D)

        def fc(z):
            # z: (B, z_dim) を想定
            return cell(z, x_sample)

        # 初期値 z0
        z0 = torch.zeros(B, z_dim, device=device)

        # ---- Anderson 反復を実行 ----
        z_final, relres = anderson(
            fc,
            z0,
            z_dim=z_dim,
            m=m,
            num_iter=num_iter,
            tol=tol,
            beta=beta
        )

    # ---- 可視化 ----
    if Show:
        plt.figure(figsize=(6,4))
        plt.semilogy(range(1, len(relres)+1), relres, marker="o")
        plt.xlabel("iteration")
        plt.ylabel("relative residual")
        plt.title("convergence on tabular sample (relative residual)")
        plt.grid(True, which="both")
        plt.tight_layout()
        plt.show()

    return relres


#------------------------------------------------------------------------------------------
def show_images(images,labels,dataset,fixed_indices):
    dataset_config = {
    'mnist':     {'img_size': 28, 'channels': 1, 'title': "MNIST Original Images"},
    'cifar-10':  {'img_size': 32, 'channels': 3,'title': "CIFAR-10 Original Images"},
    'cinic-10':  {'img_size': 32, 'channels': 3,'title': "CINIC-10 Original Images"},
    'fashion-mnist': {'img_size': 28, 'channels': 1, 'title': "Fashion-MNIST Original Images"},
    'cifar-100': {'img_size': 32, 'channels': 3, 'title': "CIFAR-100 Original Images"},
    }
    img_size = dataset_config[dataset]['img_size']
    channels = dataset_config[dataset]['channels']
    title= dataset_config[dataset]['title']

    images = images.view(images.size(0),channels,
                         img_size,img_size)
    selected_classes = list(fixed_indices.keys())

    class_names = {
        'mnist': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        'cifar-10': ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
        'cinic-10': ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
        'fashion-mnist': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    }

    images = images.cpu().numpy()
    images = images.transpose(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
    mean = np.array([0.5,0.5,0.5])
    std = np.array([0.5,0.5,0.5])
    images =  images * std + mean  # 標準化を元に戻す
    images = np.clip(images, 0, 1)  # 0-1の範囲にクリップ

    num_classes = len(selected_classes)
    fig, axes = plt.subplots(1, num_classes, figsize=(num_classes * 5, 5))
    for i, class_idx in enumerate(selected_classes):
        indices = np.where(labels == class_idx)[0]
        if len(indices) > 0:
            if class_idx in fixed_indices:
                idx = indices[fixed_indices[class_idx]]
            else:
                idx = indices[0]  # 最初のインデックスを使用
            axes[i].imshow(images[idx])
            axes[i].set_title(f"{class_names[dataset][class_idx]}")
            axes[i].axis('off')
        else:
            print(f"No images found for class {class_idx} in dataset {dataset}.")
            axes[i].axis('off')  # 該当クラスがない場合は非表示
    plt.suptitle(title)

#------------------------------------------------------------------------------------------
def final_graph_maker(file_pathes,variable_param,variable_values,memory_lis,labels,Save=False, Show=False):
    labelsize = 15
    fontsize = 25
    fmts = ['-o', '-s', '-^', '-D', '-v', '-<', '-p', '-*', '-h', '-H'] # 各モデルのプロットスタイル

    # LOSSのグラフ
    fig1, ax1 = plt.subplots(figsize=(6, 4.5))
    ax1.tick_params(axis='both', labelsize=labelsize)

    # ACCのグラフ
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    ax2.tick_params(axis='both', labelsize=labelsize)

    i = 0
    for file_path in file_pathes:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)
        All_last_ACCs_ = np.array([eval(row) for row in rows[0]])
        All_last_LOSSs_ = np.array([eval(row) for row in rows[1]]) 
        All_last_LOSSs_ = np.array(All_last_LOSSs_)
        All_last_ACCs_ = np.array(All_last_ACCs_)

        # データを2次元配列に変換（必要に応じて）
        LOSS_means = np.mean(All_last_LOSSs_, axis=1)  # 各 leverage に対する平均
        LOSS_stds = np.std(All_last_LOSSs_, axis=1)   # 各 leverage に対する標準偏差
        ACC_means = np.mean(All_last_ACCs_, axis=1)  # 各 leverage に対する平均
        ACC_stds = np.std(All_last_ACCs_, axis=1)   # 各 leverage に対する標準偏差

        ax1.errorbar(
            x=variable_values, y=LOSS_means, yerr=LOSS_stds,
            fmt=fmts[i], color=colors[i], ecolor=colors[i], capsize=5, 
            label=labels[i]
        )

        ax2.errorbar(
            x=variable_values, y=ACC_means, yerr=ACC_stds,
            fmt=fmts[i], color=colors[i], ecolor=colors[i], capsize=5,
            label=labels[i]
        )
        i += 1
        
    x_label = variable_param.capitalize()
    if variable_param == 'leverage':
        x_label = 'Compression Ratio'
    elif variable_param == 'compressed_dim':
        x_label = 'Encoder Output'

    ax1.set_xlabel(x_label, fontsize=fontsize)
    ax1.set_xticks(memory_lis)
    if variable_param == 'leverage':
        ax1.set_xticklabels([f"1:{x}" for x in memory_lis])
    elif variable_param == 'alpha':
        xticklabels = [r"$2\pi$", r"$\pi$", r"$\dfrac{\pi}{2}$", r"$\dfrac{\pi}{4}$", r"$\dfrac{\pi}{16}$"]
        # xticklabels = [r"$\dfrac{\pi}{16}$",r"$\dfrac{\pi}{32}$", r"$\dfrac{\pi}{64}$", r"$\dfrac{\pi}{128}$"]#---------------
        ax1.set_xticklabels(xticklabels)
    else:
        ax1.set_xticks(memory_lis)
    ax1.set_ylabel('LOSS', fontsize=fontsize)
    ax1.grid(True)


    ax2.set_xlabel(x_label, fontsize=fontsize)
    ax2.set_xticks(memory_lis)
    if variable_param == 'leverage':
        ax2.set_xticklabels([f"1:{x}" for x in memory_lis])
    elif variable_param == 'alpha':
        xticklabels = [r"$2\pi$", r"$\pi$", r"$\dfrac{\pi}{2}$", r"$\dfrac{\pi}{4}$", r"$\dfrac{\pi}{16}$"]
        # xticklabels = [r"$\dfrac{\pi}{16}$",r"$\dfrac{\pi}{32}$", r"$\dfrac{\pi}{64}$", r"$\dfrac{\pi}{128}$"]#---------------
        ax2.set_xticklabels(xticklabels)
    ax2.set_ylabel('Accuracy', fontsize=fontsize)
    ax2.legend(fontsize=fontsize, loc='upper left', bbox_to_anchor=(1.0, 1))
    ax2.grid(True)
    if Show:
        plt.show()
    if Save:
        tmp_loss = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        fig1.savefig(tmp_loss.name, bbox_inches='tight')
        # Accuracy用画像を保存
        tmp_acc = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        fig2.savefig(tmp_acc.name, bbox_inches='tight')

        plt.close(fig1)
        plt.close(fig2)

        return tmp_loss.name, tmp_acc.name
#-----------------------------------------------------------

def epoch_graph_maker(file_pathes, variable_param, variable_values, memory_lis,labels,Save=False, Show=False):
    fmts =  ['-o', '-s', '-^', '-D']

    fig, ax1 = plt.subplots()

    i = 0
    for file_path in file_pathes:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)
        
        # 1行目（All_loss_test）を取得してリストに変換
        All_loss_test = [eval(loss_str) for loss_str in rows[0]]
        
        epochs = len(All_loss_test[0]) 
        mean_loss = np.mean(All_loss_test, axis=0)
        std_loss = np.std(All_loss_test, axis=0)
        ax1.errorbar(
            x=range(1, epochs + 1), y=mean_loss, yerr=std_loss,
            fmt=fmts[i], color=colors[i], ecolor=colors[i], capsize=5 ,label=labels[i]
        )
        i += 1

    ax1.set_xlabel('Epochs', fontsize=15)
    ax1.set_ylabel('LOSS', fontsize=15)
    ax1.legend(fontsize=15, loc='upper right')
    ax1.grid(True)

    if Show:
        plt.show()
    if Save:
        tmp_img = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        fig.savefig(tmp_img.name, bbox_inches='tight')
        plt.close(fig)
        return tmp_img.name