import os
import platform
import sys
# sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.evaluate import final_graph_maker

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
elif system_type == "Linux":
    onedrive_path = os.path.join("/home1/konishi/Photonic_Encoder/PhotonicEncoder/result_data")
# print(f"OneDrive path: {onedrive_path}")

file_pathes = []
labels = []

#-------------------------dataset-------------------------------
dataset = 'fashion-mnist'
variable_param = 'leverage'

#-------------------------1stParameter--------------------------
labels.append('(1)PM_DEQ')
enc_type = 'PM'
cls_type = 'MLP'
variable_value = [4,8,16]  # 例: variableの値
num_times = 5  # 試行回数
experiment_name = 'DEQ7180001'

file_pathes.append(os.path.join(onedrive_path,'PhotonicEncoder_data',dataset, variable_param + '_variable', enc_type, cls_type, experiment_name, 'Final_results.csv'))

#-------------------------2ndParameter--------------------------
labels.append('(2)PM_Normal')
enc_type = 'PM'
cls_type = 'MLP'
variable_value = [4,8,16]  # 例: variableの値
num_times = 5  # 試行回数
experiment_name = 'Normal7181422'  

file_pathes.append(os.path.join(onedrive_path,'PhotonicEncoder_data',dataset, variable_param + '_variable', enc_type, cls_type, experiment_name, 'Final_results.csv'))

#-------------------------3rdParameter--------------------------
labels.append('(3)LI_Normal')
enc_type = 'LI'
cls_type = 'MLP'
variable_value = [4,8,16]  # 例: variableの値
num_times = 5  # 試行回数
experiment_name = 'Normal7212000' 

file_pathes.append(os.path.join(onedrive_path,'PhotonicEncoder_data',dataset, variable_param + '_variable', enc_type, cls_type, experiment_name, 'Final_results.csv'))
#---------------------------------------------------------------
if variable_param == 'leverage':
    if dataset in ('mnist', 'fashion-mnist') :
        variable_values = [1,2,4,8,16]
        memory_lis =[1,2,4,8,16]
    elif dataset in ('cifar-10', 'cinic-10'):
        variable_values = [1,2,3,4,6,8,12,16,24,48]
        memory_lis =[1,2,10,20,30,40,50]
    elif dataset == 'covtype':
        variable_values = [1,2,3,6,9,18,27,54]
        memory_lis =[1,2,10,20,30,40,50,60]

# variable_values = [1,2,4,8,16]
# memory_lis =[1,2,4,8,16]

final_graph_maker(file_pathes,variable_param,variable_value,memory_lis,labels,Save=False, Show=True)