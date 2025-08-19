import torch##
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
import sys
import time

import csv

from dataloader.dataloader import get_new_dataloader 
from models.IntegrationModel import Image10Classifier,DEQ_Image10Classifier,Table10Classifier,DEQ_Table10Classifier

def train_nomal(dataset,loss_func,optimizer,lr,num_times,num_try,data_train,
                         data_test,batch_size,device,max_epochs,leverage,
                         enc_type,alpha,cls_type,num_layer,fc,dropout,kernel_size
                         ):
    #---------------------------------------------
    models = {
        'mnist':Image10Classifier,
        'cifar-10':Image10Classifier,
        'cinic-10':Image10Classifier,
        'fashion-mnist':Image10Classifier,
        'covtype':Table10Classifier
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
        'adamw':torch.optim.AdamW,
        'rmsprop':torch.optim.RMSprop
    }
    #---------------------------------------------

    
    loss_train_ = []
    loss_test_ = []
    pro_time_ = []
    train_dataloader,test_dataloader = get_new_dataloader(data_train,
                                                            data_test,batch_size)
    model = models[dataset](dataset,kernel_size,leverage,
                            enc_type,alpha,cls_type,num_layer,fc,dropout,device)
    
    if dataset == 'covtype':
        counts = [211840, 283301, 35754, 2747, 9493, 17367, 20510] #covtypeのy_originの各ラベル数。
        class_w = torch.tensor(1.0 / np.sqrt(counts), 
                               dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=0.05)
    else:
        criterion = loss_funcs[loss_func]
    optimizer = optimizers[optimizer](model.parameters(), lr)

    for epoch in range(max_epochs):

        loss_train = 0
        loss_test = 0
        start_time1 = time.time() 
        i = 0
        for x,t in train_dataloader:
            x,t = x.to(device),t.to(device)
            y = model(x).to(device)
            loss = criterion(y,t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            i+=1
            sys.stderr.write('\r%d/%dth Epoch:%d/%d(%.2f%%) ' % (num_times+1,num_try, epoch+1, max_epochs,100*i/len(train_dataloader))) 
            sys.stderr.flush()
        loss_train_avg = loss_train/len(train_dataloader)
        end_time1 = time.time()
        # print('Epoch %d/%d:Time=%.2f' % (epoch+1, max_epochs, end_time1-start_time1))
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
    
    Last_loss_test = loss_test_[-1]
    Test_acc = 100 * correct / total
    return loss_train_,loss_test_,pro_time_,Last_loss_test,Test_acc,all_labels,all_preds 


def train_for_DEQ(dataset,loss_func,optimizer,lr,num_times,num_try,data_train,
                         data_test,batch_size,device,max_epochs,leverage,
                         enc_type,alpha,cls_type,num_layer,fc,dropout,kernel_size,num_iter,m,tol,beta,gamma,lam):
    #---------------------------------------------
    models = {
        'mnist':DEQ_Image10Classifier,
        'cifar-10':DEQ_Image10Classifier,
        'cinic-10':DEQ_Image10Classifier,
        'fashion-mnist':DEQ_Image10Classifier,
        'covtype':DEQ_Table10Classifier
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
        'adamw':torch.optim.AdamW,
        'rmsprop':torch.optim.RMSprop
    }
    #---------------------------------------------

    loss_train_ = []
    loss_test_ = []
    pro_time_ = []
    train_dataloader,test_dataloader = get_new_dataloader(data_train,
                                                            data_test,batch_size)
    model = models[dataset](dataset,kernel_size,leverage,
                            enc_type,alpha,cls_type,num_layer,fc,dropout,num_iter,m,tol,beta,gamma,lam,device)
    criterion = loss_funcs[loss_func]
    optimizer = optimizers[optimizer](model.parameters(), lr)

    for epoch in range(max_epochs):

        loss_train = 0
        loss_test = 0
        start_time1 = time.time() 
        i = 0
        for x,t in train_dataloader:
            x,t = x.to(device),t.to(device)
            y = model(x).to(device)
            loss = criterion(y,t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            i+=1
            sys.stderr.write('\r%d/%dth Epoch:%d/%d(%.2f%%) ' % (num_times+1,num_try, epoch+1, max_epochs,100*i/len(train_dataloader))) 
            sys.stderr.flush()

        loss_train_avg = loss_train/len(train_dataloader)
        end_time1 = time.time()
        pro_time_.append(end_time1-start_time1)
        # print('Epoch %d/%d:Time=%.2f' % (epoch+1, max_epochs, end_time1-start_time1))

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
    
    Last_loss_test = loss_test_[-1]
    Test_acc = 100 * correct / total
    return loss_train_,loss_test_,pro_time_,Last_loss_test,Test_acc,all_labels,all_preds

