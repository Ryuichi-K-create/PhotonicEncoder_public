import os
import platform
from datetime import datetime
import subprocess
import csv

now = datetime.now()
formatted_time = now.strftime("%m%d%H%M")
formatted_time = int(formatted_time)

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

#------------------------------------------------------------------------------------------
def auto_git_push(branch_name,commit_msg="Auto commit"):
    commit_msg = f"{formatted_time}_{commit_msg}"
    try:
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
        subprocess.run(["git", "push","origin",branch_name], check=True)
        print("✅ Git push 完了！")
    except subprocess.CalledProcessError as e:
        print("❌ Git操作でエラー:", e)

#--------------------------------------------------------------------------------
def save_csv(datas,variable_param,variable,num_times,dataset,enc_type,cls_type,save_type): #結果はonedriveに保存
    save_directory = os.path.join(onedrive_path,'PhotonicEncoder_data',dataset,f'{variable_param}_variable',enc_type,cls_type,str(formatted_time))
    os.makedirs(save_directory, exist_ok=True)

    if save_type == 'trial':
        file_name = f'{variable}{variable_param}_{num_times+1}th_.csv'##
    elif save_type=='mid':
        file_name = f'{variable}{variable_param}_mid.csv'
    elif save_type == 'final':
        file_name = f'Final_results.csv'
    
    full_path = os.path.join(save_directory, file_name)
    with open(full_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for data in datas:
            if isinstance(data,(int,float)):
                writer.writerow([data])
            else:
                writer.writerow(data)
    print(f"Saved at: {full_path}")

