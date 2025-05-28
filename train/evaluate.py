import os
import platform
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'
import numpy as np
import pandas as pd
import seaborn as sns
import csv
from datetime import datetime
from sklearn.metrics import confusion_matrix
import random
from dataloader.dataloader import get_new_dataloader 
from models.IntegrationModel import split_into_kernels, PMEncoder,IMEncoder,MZMEncoder,LIEncoder

home_directory = os.path.expanduser('~')
system_type = platform.system()

# デフォルトの OneDrive フォルダ名
onedrive_path = None
if system_type == "Windows":
    # Windows では環境変数が使える（MS公式な方法）
    onedrive_path = os.environ.get("OneDrive")
    if onedrive_path is None:
        # フォールバック
        onedrive_path = os.path.join(home_directory, "OneDrive")
elif system_type == "Darwin": 
    onedrive_path = os.path.join(home_directory, "Library", "CloudStorage", "OneDrive-個人用(2)")

def plot_loss_curve(train_loss,test_loss):
    plt.figure(figsize=(6,4.5))
    plt.plot(range(1,len(train_loss)+1),train_loss,label='training',color='blue')
    plt.plot(range(1,len(test_loss)+1),test_loss,label='test',color='cyan')
    plt.xlabel('Epochs')
    plt.ylabel('LOSS')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show

def plot_errorbar_losscurve(All_loss_test):
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
    #plt.ylim(1.0,2.0)
    plt.show()


def plot_confusion_matrix(true_labels,pred_labels,dataset,test_acc):

    num_labels = {
        'mnist':range(10),
        'cifar-10':range(10),
        'fashion-mnist':range(10),
        'cavtype':["Spruce/Fir","Lodgepole Pine","Ponderosa Pine",
                   "Cottonwood/Willow","Aspen","Douglas-fir","Krummholz" ]
    }
    cm = confusion_matrix(true_labels,pred_labels)
    cm = cm.astype('float')/cm.sum(axis=1,keepdims = True)
    plt.figure(figsize=(6,4.5))#8,6
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=num_labels[dataset],yticklabels=num_labels[dataset],vmin=0.0, vmax=1.0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f"Overall Correction Rate:{test_acc:.2f}%")
    plt.show()

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

def save_csv(folder,ex_name,data1,data2=None): #結果はonedriveに保存
    save_directory1 = os.path.join(onedrive_path,'Codes','PhotonicEncoder_data',folder)
    print(save_directory1)
    os.makedirs(save_directory1, exist_ok=True)
    now = datetime.now()
    formatted_time = now.strftime("%m%d%H%M")
    formatted_time = int(formatted_time)
    file_name = f'{ex_name}_{formatted_time}.csv'##
    full_path = os.path.join(save_directory1, file_name)
    with open(full_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data1)
        if data2 is not None:
            writer.writerow(data2)
    print(f"Saved at: {full_path}")

def create_table(All_test_acc,All_last_loss,All_pro_time):
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
        "": ["ACC", "LOSS", "TIME"],
        "Average": [ACC_mean, LOSS_mean, PRO_mean],
        "Best ID": [ACC_bestID, LOSS_bestID, PRO_bestID],
        "Best": [ACC_best, LOSS_best, PRO_best],
        "Worst ID": [ACC_worstID, LOSS_worstID, PRO_worstID],
        "Worst": [ACC_worst, LOSS_worst, PRO_worst],
    }
    df = pd.DataFrame(data)
    print(df)

