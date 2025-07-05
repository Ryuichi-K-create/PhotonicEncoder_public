import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from result_management.data_manager import create_result_pdf
# from train.evaluate import plot_loss_curve, plot_errorbar_losscurve, plot_confusion_matrix, plot_histograms, create_table, convergence_verify, final_graph_maker
print("-------import finished-------")
#--------------------------------------------------------
variable_param = "leverage"
params={
    'dataset': 'mnist',
    'variable_param': 'leverage',
    'enc_type': 'IM',
    'cls_type': 'MLP',
    'formatted_time': '7052014',  # 例: 'mmddyyyy'
    'leverage': [1, 2, 4],  # 例: [1, 2, 4, 8, 16]
    'num_try': 2,  # 試行回数
}

create_result_pdf(variable_param, params)





# #--------------------------------------------------------
# # フォルダパス
# folder_path = os.path.join(params['dataset'], params['variable_param'] + '_variable', params['enc_type'], params['cls_type'], params['formatted_time'])

# #trialのファイル名
# trial_file_name = f'{variable_value}{params['variable_param']}_{num_times}th_.csv'
# loss_train_, loss_test_, all_labels, all_preds, Test_acc = load_csv_data(folder_path,trial_file_name)

# print(f"Loss train: {loss_train_}")
# print(f"Loss test: {loss_test_}")
# print(f"Labels: {all_labels}")
# print(f"Predictions: {all_preds}")
# print(f"Test accuracy: {Test_acc}")



# #--------------------------------------------------------
# import platform
# from reportlab.lib.pagesizes import A4
# from reportlab.pdfgen import canvas
# from reportlab.lib.utils import ImageReader
# home_directory = os.path.expanduser('~')
# system_type = platform.system()

# # デフォルトの OneDrive フォルダ名
# onedrive_path = None
# if system_type == "Windows":
#     # Windows では環境変数が使える（MS公式な方法）
#     onedrive_path = os.environ.get("OneDrive")
#     if onedrive_path is None:
#         # フォールバック
#         onedrive_path = os.path.join(home_directory, "OneDrive")
# elif system_type == "Darwin": 
#     onedrive_path = os.path.join(home_directory, "Library", "CloudStorage", "OneDrive-個人用(2)")
# #--------------------------------------------------------
# save_dir = os.path.join(onedrive_path, 'PhotonicEncoder_data', dataset, f'{variable_param}_variable', enc_type, cls_type, formatted_time)

# loss_curve_name=plot_loss_curve(loss_train_,loss_test_,Save=True)
# confusion_name = plot_confusion_matrix(all_labels,all_preds,dataset,Test_acc,Save=True)

# save_path = os.path.join(save_dir, "result1.pdf")
# c = canvas.Canvas(save_path, pagesize=A4)
# width, height = A4

# # 画像サイズ
# img_width = 200
# img_height = 250
# gap = 40  # 画像間の隙間

# # ページ中央のx座標を計算
# center_x = width // 2
# left_img_x = center_x - img_width - gap // 2
# right_img_x = center_x + gap // 2

# img_y = height - 300  # 上からの位置（調整可）

# # タイトル
# c.setFont("Helvetica-Bold", 18)
# c.drawString(50, height - 50, "実験結果レポート")

# # ラベル
# c.setFont("Helvetica", 14)
# c.drawString(left_img_x, img_y + img_height + 1, "Loss Curve")
# c.drawString(right_img_x, img_y + img_height + 1, "Confusion Matrix")

# # 画像を横並びで中央に
# c.drawImage(ImageReader(loss_curve_name), left_img_x, img_y, width=img_width, height=img_height, preserveAspectRatio=True)
# c.drawImage(ImageReader(confusion_name), right_img_x, img_y, width=img_width, height=img_height, preserveAspectRatio=True)

# c.save()
# print(f"PDFファイルを保存しました: {save_path}")

# #--------------------------------------------------------


# #--------------------------------------------------------

# #midのファイル名
# mid_file_name = f'{variable_value}{variable_param}_mid.csv'

# All_loss_test,All_test_acc,All_last_loss,All_pro_time = load_csv_data(folder_path, mid_file_name)

# print(type(All_loss_test), All_loss_test)

# print(f"Mid Loss test: {All_loss_test}")
# print(f"Mid Acc: {All_test_acc}")
# print(f"last loss: {All_last_loss}")
# print(f"time: {All_pro_time}")

# plot_errorbar_losscurve(All_loss_test)
# create_table(All_test_acc,All_last_loss,All_pro_time)

# #finalのファイル名
# final_file_name = 'Final_results.csv'
# All_last_ACCs, All_last_LOSSs, All_TIMEs = load_csv_data(folder_path, final_file_name)

# print(f"Final last accuracies: {All_last_ACCs}")
# print(f"Final last losses: {All_last_LOSSs}")
# print(f"Final times: {All_TIMEs}")

