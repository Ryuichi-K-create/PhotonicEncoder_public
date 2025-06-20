import torch
import sys
import os
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
print(f'Using device: {device}')

from dataloader.dataloader import load_MNIST_data,load_CINIC10_data,load_CIFAR10_data,load_Fmnist_data
from train.training import train_nomal,train_for_DEQ
from train.evaluate import plot_loss_curve,plot_errorbar_losscurve,plot_confusion_matrix,plot_histograms,create_table,save_csv,convergence_verify,auto_git_push

#data---------------------------------------------
dataset = 'fashion-mnist' # 'mnist', 'cifar-10', 'cinic-10' , 'fashion-mnist'
batch_size = 100 #64 MNIST, 100 CIFAR10, 100 CINIC10
#Encoder_Model--------------------------------
enc_type = 'PM' # 'none', 'MZM', 'LI'
cls_type = 'MLP' # 'MLP' or 'CNN'
#class_model--------------------------------------
num_layer = 2
fc ='relu'
dropout = 0.0 
#learning-----------------------------------------
loss_func = 'cross_entropy'
optimizer = 'adam'
lr = 0.001
#param--------------------------------------------
num_try = 3
max_epochs = 3
leverages = [1,2,4]#,8,16] #enc is not none
kernel_size =4
#save---------------------------------------------
folder = f'Class_{dataset}_VCR'
ex_name= f'{enc_type}_{cls_type}'

data_loaders = {
    'cifar-10': load_CIFAR10_data,
    'cinic-10': load_CINIC10_data,
    'mnist': load_MNIST_data,
    'fashion-mnist':load_Fmnist_data
}

data_train,data_test = data_loaders[dataset]()
if enc_type == 'none':
    leverage = 1

results = []
All_last_LOSSs_ = []
All_last_ACCs_ = []
for leverage in leverages:
    print(f'----------------------Running with leverage: {leverage}----------------------')
    All_last_loss = []
    All_loss_test = []
    All_pro_time = []
    All_test_acc = []
    for num_times in range(num_try):

        loss_train_,loss_test_,pro_time_,Last_loss_test,Test_acc,all_labels,all_preds = train_nomal(dataset,loss_func,optimizer,lr,num_times,num_try,data_train,data_test,batch_size,device,max_epochs,leverage,enc_type,cls_type,num_layer,fc,dropout,kernel_size)

        All_loss_test.append(loss_test_)
        All_pro_time.append(sum(pro_time_))
        All_last_loss.append(Last_loss_test)
        All_test_acc.append(Test_acc)

        plot_loss_curve(loss_train_,loss_test_)
        plot_confusion_matrix(all_labels,all_preds,dataset,Test_acc)

    plot_errorbar_losscurve(All_loss_test)
    create_table(All_test_acc,All_last_loss,All_pro_time)

    All_last_LOSSs_.append(All_last_loss)
    All_last_ACCs_.append(All_test_acc)

save_csv(folder,ex_name,All_last_LOSSs_,All_last_ACCs_)


