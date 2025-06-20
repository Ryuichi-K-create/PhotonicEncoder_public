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

#--------------------------------------------------------------
root1 = os.path.join(onedrive_path,'Codes','PhotonicEncoder_data','Class_cifar-10')

file1 = os.path.join(root1, 'PM_CNN_6200040.csv')
file2 = os.path.join(root1, 'IM_CNN_6200040.csv')
file_pathes = [file1, file2]

leverages = [1,2,4,8,16]
memory_lis =[1,2,4,8,16]
labels = ['(1)Phase Modulation','(2)Intensity Modulation','(3)MZ Modulation','(4)Linear'] 

graph_maker(file_pathes, leverages, memory_lis, labels)