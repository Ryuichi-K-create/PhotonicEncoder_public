import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from result_management.data_manager import create_result_pdf
# from train.evaluate import plot_loss_curve, plot_errorbar_losscurve, plot_confusion_matrix, plot_histograms, create_table, convergence_verify, final_graph_maker
print("-------import finished-------")
#--------------------------------------------------------
variable_param = "alpha"
params={
    'dataset': 'cinic-10',  # 'mnist', 'cifar-10', 'cinic-10', 'fashion-mnist'
    'enc_type': 'PM',
    'cls_type': 'CNN',
    'formatted_time': '7161457',  # 例: 'mmddyyyy'
    variable_param: [np.pi/16,np.pi/32,np.pi/64,np.pi/128],  
    # 例: [1, 2, 4, 8, 16],[np.pi*2,np.pi, np.pi/2, np.pi/4, np.pi/8, np.pi/16]
    'num_try': 5,  # 試行回数
}

create_result_pdf(variable_param, params)

