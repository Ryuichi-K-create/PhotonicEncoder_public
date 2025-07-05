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

