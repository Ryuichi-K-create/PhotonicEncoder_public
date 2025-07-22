import os
import platform
from datetime import datetime
import subprocess
import csv
import ast
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
import numpy as np
from reportlab.lib import colors

from train.evaluate import plot_loss_curve, plot_confusion_matrix, plot_errorbar_losscurve, create_table, final_graph_maker

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
def save_csv(datas,variable_param,variable,num_times,dataset,enc_type,cls_type,save_type,experiment_name): #結果はonedriveに保存
    save_directory = os.path.join(onedrive_path,'PhotonicEncoder_data',dataset,f'{variable_param}_variable',enc_type,cls_type,str(experiment_name))
    os.makedirs(save_directory, exist_ok=True)
    if variable_param == 'alpha':
        variable = f'{variable/np.pi:.3f}π'

    if save_type == 'trial':
        file_name = f'{variable}{variable_param}_{num_times+1}th_.csv'##
    elif save_type=='mid':
        file_name = f'{variable}{variable_param}_mid.csv'

    elif save_type =='relres':
        file_name = f'{variable}{variable_param}_relres.csv'
    
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
        print(f"Loading CSV file: {full_path}")
        with open(full_path, 'r', newline='') as file:
            reader = csv.reader(file)
            rows = list(reader)
            datas = []
            for row in rows:
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
                        converted = [int(float(x)) for x in row]
                    datas.append(converted if len(converted) > 1 else converted[0])

                except Exception:
                    if all(isinstance(x, str) and x.startswith('[') and x.endswith(']') for x in row):
                        try:
                            converted = [ast.literal_eval(x) for x in row]
                            datas.append(converted)
                        except Exception:
                            datas.append(row[0])
                    else:
                        datas.append(row if len(row) > 1 else row[0])
            return tuple(datas)
    except Exception as e:
        print(f"Error loading CSV file {file_name}: {e}")
        return tuple(None for _ in range(len(rows))) if 'rows' in locals() else ()

#--------------------------------------------------------------------------------

def save_experiment_report(variable_param, params,experiment_name='Normal'):
    # 保存先ディレクトリの構築
    save_directory = os.path.join(onedrive_path,'PhotonicEncoder_data',params['dataset'],
                                  f'{variable_param}_variable', params['enc_type'], params['cls_type'], str(experiment_name))
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

#--------------------------------------------------------------------------------
def create_result_pdf(variable_param, params,experiment_name='Normal',Show=False):
    folder_path = os.path.join(onedrive_path, 'PhotonicEncoder_data', params['dataset'],f"{variable_param}_variable", params['enc_type'], params['cls_type'], str(experiment_name))
    file_name = 'experiment_report.pdf'
    c = canvas.Canvas(f"{folder_path}/{file_name}", pagesize=A4)
    width, height = A4

    # --- レイアウト設定 ---
    IMG_WIDTH = 200
    IMG_HEIGHT = 180
    H_GAP = 20  # 水平方向のギャップ
    V_GAP = 40  # 垂直方向のギャップ
    TOP_MARGIN = 50
    BOTTOM_MARGIN = 50
    
    center_x = width / 2
    left_x = center_x - IMG_WIDTH - H_GAP / 2
    right_x = center_x + H_GAP / 2
    
    current_y = height - TOP_MARGIN

    def new_page_check(required_height):
        nonlocal current_y, c
        if current_y - required_height < BOTTOM_MARGIN:
            c.showPage()
            current_y = height - TOP_MARGIN
            return True
        return False

    # --- PDF生成開始 ---
    
    # 全体タイトル
    c.setFont("Times-Roman", 20)
    c.drawCentredString(center_x, current_y, "Experiment Result Report")
    current_y -= 40

    # 可変パラメータごとのループ
    for variable in params[variable_param]:
        if variable_param == 'alpha':
            variable = f'{variable/np.pi:.3f}π'
        
        # セクションタイトル
        if new_page_check(30):
             c.drawCentredString(center_x, current_y, "Experiment Result Report (Cont.)")
             current_y -= 40
        c.setFont("Times-Roman", 16)
        c.drawString(left_x, current_y, f"Variable: {variable_param} = {variable}")
        current_y -= 30

        # 試行ごとのループ
        for num_times in range(params['num_try']):
            if new_page_check(IMG_HEIGHT + 20): # ラベル分+画像
                c.setFont("Times-Roman", 16)
                c.drawString(left_x, current_y, f"Variable: {variable_param} = {variable} (Cont.)")
                current_y -= 30

            trial_file_name = f"{variable}{variable_param}_{num_times+1}th_.csv"
            loss_train_, loss_test_, all_labels, all_preds, Test_acc = load_csv_data(folder_path, trial_file_name)
            
            c.setFont("Times-Roman", 12)
            c.drawString(left_x, current_y, f"Trial #{num_times+1}: Loss Curve")
            c.drawString(right_x, current_y, f"Trial #{num_times+1}: Confusion Matrix")
            current_y -= (IMG_HEIGHT + 15)

            loss_curve_name = plot_loss_curve(loss_train_, loss_test_, Save=True,Show=Show)
            confusion_name = plot_confusion_matrix(all_labels, all_preds, params['dataset'], Test_acc, Save=True,Show=Show)
            c.drawImage(ImageReader(loss_curve_name), left_x, current_y, width=IMG_WIDTH, height=IMG_HEIGHT, preserveAspectRatio=True)
            c.drawImage(ImageReader(confusion_name), right_x, current_y, width=IMG_WIDTH, height=IMG_HEIGHT, preserveAspectRatio=True)
            current_y -= V_GAP

        # 中間結果
        if new_page_check(IMG_HEIGHT + 20):
            c.setFont("Times-Roman", 16)
            c.drawString(left_x, current_y, f"Variable: {variable_param} = {variable} (Cont.)")
            current_y -= 30

        mid_file_name = f"{variable}{variable_param}_mid.csv"
        All_loss_test, All_test_acc, All_last_loss, All_pro_time = load_csv_data(folder_path, mid_file_name)
        
        # --- 中間結果の描画 ---
        
        # ラベル
        c.setFont("Times-Roman", 12)
        c.drawString(left_x, current_y, "Average Loss Curve")
        c.drawString(right_x, current_y, "Statistics Summary")
        current_y -= 15 # ラベルとコンテンツの間のスペース

        # Errorbar Loss
        error_loss_name = plot_errorbar_losscurve(All_loss_test, Save=True,Show=Show)
        
        # Table
        df = create_table(All_test_acc, All_last_loss, All_pro_time,Save=True,Show=Show)
        
        table_h = 0
        if df is not None:
            table_data = [df.columns.tolist()] + df.values.tolist()
            table = Table(table_data, colWidths=[40, 48, 40, 40, 40, 40] )
            table.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
                ('FONTSIZE', (0,0), (-1,-1), 8),
                ('GRID', (0,0), (-1,-1), 0.5, 'black'),
            ]))
            _ , table_h = table.wrapOn(c, 0, 0)
        
        # y座標の計算 (画像とテーブルの高さを考慮)
        content_y = current_y - max(IMG_HEIGHT, table_h)

        # Errorbar Loss Curve を描画
        c.drawImage(ImageReader(error_loss_name), left_x, content_y, width=IMG_WIDTH, height=IMG_HEIGHT, preserveAspectRatio=True)

        # Table を描画
        if df is not None:
            table.drawOn(c, right_x-25, content_y + max(IMG_HEIGHT, table_h) - table_h) # テーブルを垂直方向中央に配置
        else:
            c.setFont("Times-Roman", 10)
            c.drawString(right_x, content_y + IMG_HEIGHT/2, "(No data to display)")

        current_y = content_y - V_GAP # y座標を更新

    if variable_param != 'none':
        # --- 最終結果グラフの追加 ---
        c.showPage() # 新しいページを開始
        current_y = height - TOP_MARGIN

        c.setFont("Times-Roman", 20)
        c.drawCentredString(center_x, current_y, "Final Result Graph")
        current_y -= 40

        memory_lis = params[variable_param]
        #----leverageのリストを設定----
        if params['dataset'] in ('mnist', 'fashion-mnist') and variable_param == 'leverage':
            memory_lis =[1,2,4,8,16]
        elif params['dataset'] in ('cifar-10', 'cinic-10') and variable_param == 'leverage':
            memory_lis =[1,2,10,20,30,40,50]
        elif params['dataset'] == 'covtype' and variable_param == 'leverage':
            memory_lis =[1,2,10,20,30,40,50,60]
        #----alphaのリストを設定----
        elif variable_param == 'alpha':
            memory_lis = [np.pi*2,np.pi, np.pi/2, np.pi/4, np.pi/16]
            # memory_lis = [np.pi/16,np.pi/32,np.pi/64,np.pi/128]
        
        #gammaのリストを設定
        elif variable_param == 'gamma':
            memory_lis = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        file_path = os.path.join(folder_path, 'Final_results.csv')
        final_loss_name, final_acc_name = final_graph_maker([file_path], variable_param,params[variable_param], memory_lis, 'Photonic Encoder', Save=True,Show=Show)

        # ラベルを描画
        c.setFont("Times-Roman", 12)
        c.drawString(left_x, current_y, "Final Loss Graph")
        c.drawString(right_x, current_y, "Final Accuracy Graph")
        current_y -= (IMG_HEIGHT + 15)

        # 画像を描画
        if final_loss_name and os.path.exists(final_loss_name):
            c.drawImage(ImageReader(final_loss_name), left_x, current_y, width=IMG_WIDTH, height=IMG_HEIGHT, preserveAspectRatio=True)
        
        if final_acc_name and os.path.exists(final_acc_name):
            c.drawImage(ImageReader(final_acc_name), right_x, current_y, width=IMG_WIDTH, height=IMG_HEIGHT, preserveAspectRatio=True)

    c.save()
    print(f"PDFファイルを保存しました: {folder_path}/{file_name}")