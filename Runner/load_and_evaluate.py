import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from result_management.data_manager import load_csv_data
from train.evaluate import plot_loss_curve, plot_errorbar_losscurve, plot_confusion_matrix, plot_histograms, create_table, convergence_verify, graph_maker

dataset = 'mnist'
variable_param = 'leverage'
enc_type = 'IM'
cls_type = 'MLP'
formatted_time = '7052014'  # 例: 'mmddyyyy'
variable_value = 2  # 例: variableの値
num_times = 1  # 試行回数
#--------------------------------------------------------
# フォルダパス
folder_path = os.path.join(dataset, variable_param + '_variable', enc_type, cls_type, formatted_time)

#trialのファイル名
trial_file_name = f'{variable_value}{variable_param}_{num_times}th_.csv'
loss_train_, loss_test_, all_labels, all_preds, Test_acc = load_csv_data(folder_path,trial_file_name)

print(f"Loss train: {loss_train_}")
print(f"Loss test: {loss_test_}")
print(f"Labels: {all_labels}")
print(f"Predictions: {all_preds}")
print(f"Test accuracy: {Test_acc}")

plot_loss_curve(loss_train_,loss_test_)
plot_confusion_matrix(all_labels,all_preds,dataset,Test_acc)
#--------------------------------------------------------

#midのファイル名
mid_file_name = f'{variable_value}{variable_param}_mid.csv'

All_loss_test,All_test_acc,All_last_loss,All_pro_time = load_csv_data(folder_path, mid_file_name)

print(type(All_loss_test), All_loss_test)

print(f"Mid Loss test: {All_loss_test}")
print(f"Mid Acc: {All_test_acc}")
print(f"last loss: {All_last_loss}")
print(f"time: {All_pro_time}")

plot_errorbar_losscurve(All_loss_test)
create_table(All_test_acc,All_last_loss,All_pro_time)

#finalのファイル名
final_file_name = 'Final_results.csv'
All_last_ACCs, All_last_LOSSs, All_TIMEs = load_csv_data(folder_path, final_file_name)

print(f"Final last accuracies: {All_last_ACCs}")
print(f"Final last losses: {All_last_LOSSs}")
print(f"Final times: {All_TIMEs}")


# leverages = [1,2,4]
# memory_lis =[1,2,4]
# labels = ['(1)Phase Modulation','(2)Intensity Modulation','(3)MZ Modulation','(4)Linear'] 
# import platform


# home_directory = os.path.expanduser('~')
# system_type = platform.system()

# デフォルトの OneDrive フォルダ名
# onedrive_path = None
# if system_type == "Windows":
#     # Windows では環境変数が使える（MS公式な方法）
#     onedrive_path = os.environ.get("OneDrive")
#     if onedrive_path is None:
#         # フォールバック
#         onedrive_path = os.path.join(home_directory, "OneDrive")
# elif system_type == "Darwin": 
#     onedrive_path = os.path.join(home_directory, "Library", "CloudStorage", "OneDrive-個人用(2)")

# file_path1 = os.path.join(onedrive_path, 'PhotonicEncoder_data', dataset, f'{variable_param}_variable', enc_type, cls_type, str(formatted_time), 'Final_results.csv')
# file_pathes = [file_path1] 
# graph_maker(file_pathes,leverages,memory_lis,labels)