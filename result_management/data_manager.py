import os
import platform
from datetime import datetime
import subprocess
import csv
import ast

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

#--------------------------------------------------------------------------------
def load_csv_data(folder_path,file_name):
    try:
        full_path = os.path.join(onedrive_path, 'PhotonicEncoder_data', folder_path, file_name)
        with open(full_path, 'r', newline='') as file:
            reader = csv.reader(file)
            rows = list(reader)
            datas = []
            for row in rows:
                print(f"Evaluating: {row}")
                # 空要素除去
                row = [x for x in row if x and x.strip()]
                if not row:
                    datas.append([])
                    continue
                # すべてfloat変換できるか
                try:
                    converted = [float(x) for x in row]
                    # すべてint変換できるか
                    if all(float(x).is_integer() for x in row):
                        print(f"Evaluating1: {row}")
                        converted = [int(float(x)) for x in row]
                    datas.append(converted if len(converted) > 1 else converted[0])

                except Exception:
                    if all(isinstance(x, str) and x.startswith('[') and x.endswith(']') for x in row):
                        print(f"Evaluating2: {row}")
                        try:
                            converted = [ast.literal_eval(x) for x in row]
                            datas.append(converted)
                        except Exception:
                            print(f"Evaluating3: {row}")
                            datas.append(row[0])
                    else:
                        print(f"Evaluating4: {row}")
                        datas.append(row if len(row) > 1 else row[0])
                print(f"Final Evaluating: {row}")
            return tuple(datas)
    except Exception as e:
        print(f"Error loading data from {folder_path}: {e}")
        return tuple(None for _ in range(len(rows))) if 'rows' in locals() else ()

#--------------------------------------------------------------------------------

def save_experiment_report(variable_param, params):
    """
    NN実験のパラメータを日本語でわかりやすくまとめ、指定フォルダにreport.txtとして保存する関数
    Args:
        variable_param (str): 可変パラメータ名
        params (dict): 実験パラメータ辞書
        folder_params (dict): 保存先フォルダ情報（dataset, enc_type, cls_type）
    """
    # 保存先ディレクトリの構築
    save_directory = os.path.join(onedrive_path,'PhotonicEncoder_data',params['dataset'],
                                  f'{variable_param}_variable', params['enc_type'], params['cls_type'], str(formatted_time))
    os.makedirs(save_directory, exist_ok=True)
    report_path = os.path.join(save_directory, 'experiment_report.txt')

    # 日本語でパラメータをまとめる
    lines = []
    lines.append('【ニューラルネットワーク実験パラメータ報告書】\n')
    lines.append(f'作成日時: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
    lines.append(f'可変パラメータ: {variable_param}\n')
    lines.append('--- 実験パラメータ一覧 ---')
    for k, v in params.items():
        lines.append(f'{k}: {v}')
    lines.append(f'保存ディレクトリ: {save_directory}')

    # ファイルに保存
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"実験パラメータ報告書を保存しました: {report_path}")

# 例: 実行
# save_experiment_report(variable_param, params, folder_params)