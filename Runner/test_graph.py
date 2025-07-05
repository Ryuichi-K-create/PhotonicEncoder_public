import os
import platform
import sys
# sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.evaluate import graph_maker

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

print(f"OneDrive path: {onedrive_path}")

file_pathes = []
labels = []

#-------------------------dataset-------------------------------
dataset = 'fashion-mnist'
variable_param = 'leverage'

#-------------------------1stParameter--------------------------
labels.append('(1)Phase Modulation')
enc_type = 'PM'
cls_type = 'MLP'
variable_value = 2  # 例: variableの値
num_times = 1  # 試行回数
formatted_time = '7060038'  #[PM,IM,MZ,LI]の順番で保存される

file_pathes.append(os.path.join(onedrive_path,'PhotonicEncoder_data',dataset, variable_param + '_variable', enc_type, cls_type, formatted_time, 'Final_results.csv'))

#-------------------------2ndParameter--------------------------
labels.append('(2)Intensity Modulation')
enc_type = 'IM'
cls_type = 'MLP'
variable_value = 2  # 例: variableの値
num_times = 1  # 試行回数
formatted_time = '7060045'  #[PM,IM,MZ,LI]の順番で保存される

file_pathes.append(os.path.join(onedrive_path,'PhotonicEncoder_data',dataset, variable_param + '_variable', enc_type, cls_type, formatted_time, 'Final_results.csv'))

#-------------------------3rdParameter--------------------------


if dataset in ('mnist', 'fashion-mnist'):
    leverages = [1,2,4,8,16]
    memory_lis =[1,2,4,8,16]
elif dataset in ('cifar-10', 'cinic-10'):
    leverages = [1,2,3,4,6,8,12,16,24,48]
    memory_lis =[1,2,10,20,30,40,50]
elif dataset == 'covtype':
    leverages = [1,2,3,6,9,18,27,54]
    memory_lis =[1,2,10,20,30,40,50,60]

# leverages = [1,2,4,8,16]
# memory_lis =[1,2,4,8,16]

graph_maker(file_pathes, leverages, memory_lis, labels)