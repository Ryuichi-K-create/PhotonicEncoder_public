import torch
import sys
import os
import numpy as np
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.abspath("..")) 
from datetime import datetime

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
print(f'Using device: {device}')

from dataloader.dataloader import load_MNIST_data,load_CINIC10_data,load_CIFAR10_data,load_Fmnist_data,load_Covtype_data
from train.training import train_for_DEQ
from train.evaluate import convergence_verify,convergence_verify_tabular
from result_management.data_manager import save_csv,save_experiment_report,create_result_pdf
now = datetime.now()
formatted_time = now.strftime("%m%d%H%M")
formatted_time = int(formatted_time)
print(f'-----Formatted time: {formatted_time} -----')
#-----------------------------------------------------------------
experiment_type = "DEQ"
experiment_name = f"{experiment_type}{formatted_time}"

variable_param = "none" #ここで設定した項目は配列にすること(none,leverage,alpha)
save = False

params = {
    'none':[0], #variable_param=noneの際は1回だけ繰り返す
    #data---------------------------------------------
    'dataset': 'covtype', # 'mnist', 'cifar-10', 'cinic-10' , 'fashion-mnist'
    'batch_size': 100, #64 MNIST, 100 CIFAR10, 100 CINIC10

    #Encoder_Model--------------------------------
    'enc_type': 'PM', # 'none', 'MZM', 'LI'
    'alpha': np.pi/2, 
    #位相変調機の感度[np.pi*2,np.pi, np.pi/2, np.pi/4, np.pi/8, np.pi/16],pi:-π~π
    #class_model--------------------------------------
    'cls_type': 'MLP', # 'MLP' or 'CNN'
    'num_layer': 2,
    'fc': 'relu',
    'dropout': 0.0,

    #learning-----------------------------------------
    'loss_func': 'cross_entropy',
    'optimizer': 'adam',
    'lr': 0.001,

    #param--------------------------------------------
    'num_try': 3,
    'max_epochs': 10,
    'leverage': 9, #mnist:[1,2,4,8,16],cinic:[1,2,3,4,6,8,12,16,24,48] enc is not none
    'kernel_size': 4,

    #anderson param-----------------------------------
    'm': 5,
    'lam': 1e-5, 
    'num_iter': 50,
    'tol': 1e-4,  #早期終了条件
    'beta': 1.0,
    'gamma' : 0 #SNLinearRelaxのgamma値(使わない)
}
#save---------------------------------------------
folder_params = {k: params[k] for k in ['dataset', 'enc_type', 'cls_type']}
if save:
    save_experiment_report(variable_param, params,experiment_name=experiment_name)

data_loaders = {
    'covtype': load_Covtype_data,
    'cifar-10': load_CIFAR10_data,
    'cinic-10': load_CINIC10_data,
    'mnist': load_MNIST_data,
    'fashion-mnist':load_Fmnist_data
}

data_train,data_test = data_loaders[params["dataset"]]()

if params["enc_type"] == 'none':
    params["leverage"] = 1
results = []
All_last_LOSSs_ = []
All_last_ACCs_ = []
All_TIMEs_ = []

for variable in params[variable_param]: #variable:leverage,alpha
    print(f'----------------------Running with {variable_param}: {variable}----------------------')
#-----------------------------------------------------
    Relres_ = []
    Unresovable = 0
    k = 1000
    Show_rel = False
    for i in range(k):
        relres = convergence_verify_tabular(params,gamma=params['gamma'],data_train=data_train,data_test=data_test,device=device,Show=Show_rel)
        Relres_.append(len(relres))
        if len(relres) > 40:
            Unresovable += 1
        sys.stderr.write(f"\rIteration {i+1}/{k} completed. Current length: {len(relres)}")
        sys.stdout.flush()
    time.sleep(1)
    print(f"Average number of iterations: {np.mean(Relres_)}")
    print(f"Unresolvable cases: {Unresovable}")
#-----------------------------------------------------
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
        loss_train_,loss_test_,pro_time_,Last_loss_test,Test_acc,all_labels,all_preds = train_for_DEQ(**params_for_train,data_train=data_train,data_test=data_test)

        All_loss_test.append(loss_test_)
        All_pro_time.append(sum(pro_time_))
        All_last_loss.append(Last_loss_test)
        All_test_acc.append(Test_acc)
        if save:
            datas = [loss_train_,loss_test_,all_labels,all_preds,Test_acc]
            save_csv(datas,variable_param,variable,num_times,**folder_params,save_type='trial',experiment_name=experiment_name)
        print(f'Test Accuracy: {Test_acc}')
        # plot_loss_curve(loss_train_,loss_test_)
        # plot_confusion_matrix(all_labels,all_preds,params["dataset"],Test_acc)

    if save:
        datas = [All_loss_test,All_test_acc,All_last_loss,All_pro_time]
        save_csv(datas,variable_param,variable,num_times,**folder_params,save_type='mid',experiment_name=experiment_name)

        datas = [Relres_,np.mean(Relres_),Unresovable]
        save_csv(datas,variable_param,variable,num_times,**folder_params,save_type='relres',experiment_name=experiment_name)
    print(f"Test Accuracy:{Test_acc:.2f}")
    # plot_errorbar_losscurve(All_loss_test)
    # create_table(All_test_acc,All_last_loss,All_pro_time)

    All_last_ACCs_.append(All_test_acc)
    All_last_LOSSs_.append(All_last_loss)
    All_TIMEs_.append(All_pro_time)

if variable_param != 'none'and save:
    datas = [All_last_ACCs_,All_last_LOSSs_,All_TIMEs_]
    save_csv(datas,variable_param,variable,num_times,**folder_params,save_type='final',experiment_name=experiment_name) #最終保存

if save:
    create_result_pdf(variable_param, params, experiment_name=experiment_name)