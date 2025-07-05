import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from result_management.data_manager import load_trial_data

file_path = 'PhotonicEncoder_data/mnist/leverage_variable/IM/MLP/7052014/1leverage_1th_.csv'

loss_train_, loss_test_, all_labels, all_preds, Test_acc = load_trial_data(file_path)

print(f"Loss train: {loss_train_}")
print(f"Loss test: {loss_test_}")
print(f"Labels: {all_labels}")
print(f"Predictions: {all_preds}")
print(f"Test accuracy: {Test_acc}")