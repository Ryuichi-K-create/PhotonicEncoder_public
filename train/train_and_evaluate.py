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
