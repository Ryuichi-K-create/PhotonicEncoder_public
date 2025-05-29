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
from models.IntegrationModel import Image10Classifier,DEQ_Image10Classifier

def train_nomal(dataset,loss_func,optimizer,lr,num_times,num_try,data_train,
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
                            enc_type,cls_type,num_layer,fc,device)
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
                         enc_type,cls_type,num_layer,fc,kernel_size,num_iter
                         ):
    #---------------------------------------------
    models = {
        'mnist':DEQ_Image10Classifier,
        'cifar-10':DEQ_Image10Classifier,
        'fashion-mnist':DEQ_Image10Classifier
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
                            enc_type,cls_type,num_layer,fc,num_iter,device)
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
