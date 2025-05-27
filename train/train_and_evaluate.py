import torch##
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt 
import sys
import time
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import platform
from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'

from dataloader.dataloader import get_new_dataloader 
from models.IntegrationModel import Image10Classifier

def plot_loss_curve(train_loss,test_loss):
    plt.figure(figsize=(8,6))
    plt.plot(range(1,len(train_loss)+1),train_loss,label='training',color='blue')
    plt.plot(range(1,len(test_loss)),test_loss,label='test',color='cyan')
    plt.xlabel('Epochs')
    plt.ylabel('LOSS')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show

def plot_confusion_matrix(true_labels,pred_labels,dataset,test_acc):

    num_labels = {
        'mnist':range(10),
        'cifar-10':range(10),
        'fashion-mnist':range(10),
        'cavtype':["Spruce/Fir","Lodgepole Pine","Ponderosa Pine",
                   "Cottonwood/Willow","Aspen","Douglas-fir","Krummholz" ]
    }
    cm = confusion_matrix(true_labels,pred_labels)
    cm = cm.astype('float')/cm.sum(axis=1,keeping = True)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=num_labels[dataset],yticklabels=num_labels[dataset],vmin=0.0, vmax=1.0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f"Overall Correction Rate:{test_acc:.2f}%")
    plt.show()

def plot_histograms(x, model, kernel_size, batch_size, channels, img_size):
    x = x.view(batch_size, channels, img_size, img_size)
    x_splitted = model.split(x, kernel_size)
    x_in_flat = x_splitted.reshape(-1).detach().cpu().numpy()
    x_encoded = model.encoder(x_splitted)
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


def train_and_classifier(dataset,loss_func,optimizer,lr,num_try,data_train,
                         data_test,batch_size,device,max_epochs,leverage,
                         enc_type,cls_type,num_layer,fc,kernel_size=4,
                         ):
    #---------------------------------------------
    models = {
        'mnist':Image10Classifier,
        'cifar-10':Image10Classifier,
        'fashion-mnist':Image10Classifier
    }
    #---------------------------------------------
    loss_funcs = {
        'cross_entropy':nn.CrossEntropyLoss(),
        'mse':nn.MSELoss(),
        'bce':nn.BCELoss()
    }
    #---------------------------------------------
    optimizers = {
        'adam':torch.optim.Adam,
        'sgd':torch.optim.SGD,
        'adamw':torch.optim.Adamw,
        'rmsprop':torch.optim.RMSprop
    }
    #---------------------------------------------
    All_last_loss = []
    All_loss_test = []
    All_pro_time = []
    All_test_acc = []

    for num_times in range(num_try):
        loss_train_ = []
        loss_test_ = []
        pro_time_ = []
        train_dataloader,test_dataloader = get_new_dataloader(data_train,
                                                              data_test,batch_size)
        model = models[dataset](dataset,kernel_size,leverage,
                                enc_type,cls_type,num_layer,fc)
        criterion = loss_funcs[loss_func]
        optimizer = optimizers[optimizer](model.parameters(), lr)

        for epoch in range(max_epochs):
            sys.stderr.write('\r%d/%dth Time Epoch: %d/%d' % (num_times+1,num_try, epoch+1, max_epochs)) 
            sys.stderr.flush()
        
            loss_train = 0
            loss_test = 0
            start_time1 = time.time() 
            for x,t in train_dataloader:
                x,t = x.to(device),t.to(device)
                y = model(x).to(device)
                loss = criterion(y,t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_train += loss.item()

            loss_train_avg = loss_train/len(loss_train)
            end_time1 = time.time()
            pro_time_.append(end_time1-start_time1)

            model.eval()
            with torch.no_grad():
                all_preds = []
                all_labels = []
                correct = 0
                total = 0
                for x,t in test_dataloader:
                    x,t = x.to(device),t.to(device)
                    y = model(x)
                    loss = criterion(y,t)
                    loss_test += loss.item()
                    total += t.size(0)
                    _,predicted = torch.max(y,dim=1)
                    correct += (predicted==t).sum().item()
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(t.cpu().numpy())

            loss_test_avg = loss_test / len(test_dataloader)
            loss_train_.append(loss_train_avg)
            loss_test_.append(loss_test_avg)
        
        All_loss_test.append(loss_test_)
        All_pro_time.append(pro_time_)
        Last_loss_test = loss_test_[-1]
        All_last_loss.append(Last_loss_test)
        Test_acc = 100 * correct / total
        All_test_acc.append(Test_acc)

    return All_loss_test,All_pro_time,Last_loss_test,All_last_loss,All_test_acc
