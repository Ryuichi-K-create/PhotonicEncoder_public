import torch
import sys
import os
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
print(f'Using device: {device}')

from dataloader.dataloader import load_compressed_Fmnist_data,load_csv_Fmnist_data
from train.training import train_nomal
from train.evaluate import plot_loss_curve,plot_errorbar_losscurve,plot_confusion_matrix,create_table
from result_management.data_manager import save_csv,save_experiment_report,create_result_pdf
now = datetime.now()
formatted_time = now.strftime("%m%d%H%M")
formatted_time = int(formatted_time)
# formatted_time = 11190120
print(f'-----Formatted time: {formatted_time} -----')
#-----------------------------------------------------------------
experiment_type = "fft_sim" # 'fft_phyz' or 'fft_sim'

data_id = 0  # fft_phyzのみ　0~5
experiment_name = f"{experiment_type}{formatted_time}_No{data_id}"
variable_param = "none" #ここで設定した項目は配列にすること(none,leverage,alpha)
save = False 
show = False 

params = {
    'none':[0], #variable_param=noneの際は1回だけ繰り返す
    #data---------------------------------------------
    'dataset': 'fashion-mnist', # 'mnist', 'cifar-10', 'cinic-10' , 'fashion-mnist'
    'batch_size': 100, #64 MNIST, 100 CIFAR10, 100 CINIC10

    #Encoder_Model--------------------------------
    'enc_type': 'PM', # 'none', 'MZM', 'LI'
    'alpha': np.pi/2, 
    #位相変調機の感度[np.pi*2,np.pi, np.pi/2, np.pi/4, np.pi/8, np.pi/16],pi:-π~π
    #class_model--------------------------------------
    'cls_type': 'MLP', # 'MLP' or 'CNN'
    'num_layer': 2,
    'fc': 'relu', #num_layer>=2のときのみ有効
    'dropout': 0.0,

    #learning-----------------------------------------
    'loss_func': 'cross_entropy',
    'optimizer': 'adam',
    'lr': 0.001,

    #param--------------------------------------------
    'num_try': 5,
    'max_epochs': 50,
    'leverage': 0, #mnist:[1,2,4,8,16],cinic:[1,2,3,4,6,8,12,16,24,48](fft特徴量版では設定しない)
    'kernel_size': 0, #(fft特徴量版では設定しない)
    #fft----------------------------------------------
    'fft_dim': 32, # FFT特徴量の次元数
    'enc_out': 17, # FFT出力の次元数
    'compressed_dim': 17#list(range(17,0,-1)) # 圧縮後の次元数 
}
#save---------------------------------------------
folder_params = {k: params[k] for k in ['dataset', 'enc_type', 'cls_type']}
if save:
    save_experiment_report(variable_param, params,experiment_name=experiment_name)


if experiment_type == "fft_sim":
    data_train,data_test = load_csv_Fmnist_data()
elif experiment_type == "fft_phyz":
    data_train,data_test = load_compressed_Fmnist_data(data_id=data_id)
else:
    raise KeyError("experiment_type is invalid.")

if params["enc_type"] == 'none':    
    leverage = 1
results = []
All_last_LOSSs_ = []
All_last_ACCs_ = []
All_TIMEs_ = []

for variable in params[variable_param]: #variable:leverage,alpha 
    print(f'----------------------ID:{data_id} Running with {variable_param}: {variable}----------------------')
    All_last_loss = []
    All_loss_test = []
    All_pro_time = []
    All_test_acc = []

    for num_times in range(params['num_try']):
        params_for_train = {k: v for k,v in params.items() if k not in ('none',variable_param)}#配列を除外

        if variable_param != 'none': #leverageやalpha可変のとき
            params_for_train.update({'num_times': num_times, variable_param: variable,'device': device})
        else: #パラメータ不変のとき
            params_for_train.update({'num_times': num_times,'device': device})
        
        #-----------training-----------
        loss_train_,loss_test_,pro_time_,Last_loss_test,Test_acc,all_labels,all_preds = train_nomal(**params_for_train,data_train=data_train,data_test=data_test,ex_type=experiment_type)

        All_loss_test.append(loss_test_)
        All_pro_time.append(sum(pro_time_))
        All_last_loss.append(Last_loss_test)
        All_test_acc.append(Test_acc)
        if save:
            datas = [loss_train_,loss_test_,all_labels,all_preds,Test_acc]
            save_csv(datas,variable_param,variable,num_times,**folder_params,save_type='trial',experiment_name=experiment_name)
        print(f"Test Accuracy:{Test_acc:.2f}")
        if show:
            plot_loss_curve(loss_train_,loss_test_,Save=save,Show=show)
            plot_confusion_matrix(all_labels,all_preds,params["dataset"],Test_acc,Save=save,Show=show)

    datas = [All_loss_test,All_test_acc,All_last_loss,All_pro_time]

    if save:
        save_csv(datas,variable_param,variable,num_times,**folder_params,save_type='mid',experiment_name=experiment_name)
    if show:  
        plot_errorbar_losscurve(All_loss_test,Save=save,Show=show)

    create_table(All_test_acc,All_last_loss,All_pro_time,Save=save,Show=True)

    All_last_ACCs_.append(All_test_acc)
    All_last_LOSSs_.append(All_last_loss)
    All_TIMEs_.append(All_pro_time)

if variable_param != 'none'and save:
    datas = [All_last_ACCs_,All_last_LOSSs_,All_TIMEs_]
    save_csv(datas,variable_param,variable,num_times,**folder_params,save_type='final',experiment_name=experiment_name) #最終保存

if save:
    create_result_pdf(variable_param, params,experiment_name=experiment_name)
