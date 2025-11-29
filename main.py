import sys
import os
import torch
import numpy as np
from datetime import datetime

# Add project root to system path to enable absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader.dataloader import (
    load_MNIST_data,
    load_CINIC10_data,
    load_CIFAR10_data,
    load_Fmnist_data,
    load_Covtype_data
)
from train.training import train_nomal as train_model
from train.evaluate import (
    plot_loss_curve,
    plot_errorbar_losscurve,
    plot_confusion_matrix,
    create_table
)
from result_management.data_manager import (
    save_csv,
    save_experiment_report,
    create_result_pdf
)

def get_device():
    """Check for GPU availability and return the appropriate device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    return device

def main():
    # --- Configuration ---
    device = get_device()
    
    # Experiment settings
    experiment_type = "normal"  # Options: 'normal', 'fft', 'deq'
    timestamp = datetime.now().strftime("%m%d%H%M")
    experiment_name = f"{experiment_type}{timestamp}"
    
    # Variable parameter for parameter sweep experiments
    # Options: 'none', 'leverage', 'alpha'
    # If set to something other than 'none', the corresponding value in `params` must be a list.
    variable_param = "none" 
    
    save_results = False
    show_plots = True

    # Hyperparameters and Model Settings
    params = {
        # Variable parameters (list if variable_param is set, else list with single value)
        'none': [0], 
        
        # Dataset
        'dataset': 'cifar-10',  # 'mnist', 'cifar-10', 'cinic-10', 'fashion-mnist'
        'batch_size': 100,      # Recommended: 64 for MNIST, 100 for CIFAR10/CINIC10

        # Encoder Model
        'enc_type': 'none',     # 'none', 'MZM', 'LI'
        'alpha': np.pi / 2,     # Phase modulator sensitivity: [2*pi, pi, pi/2, pi/4, ...]

        # Classifier Model
        'cls_type': 'MLP',      # 'MLP' or 'CNN'
        'num_layer': 1,
        'fc': 'relu',           # Activation function (valid when num_layer >= 2)
        'dropout': 0.0,

        # Training
        'loss_func': 'cross_entropy',
        'optimizer': 'adam',
        'lr': 0.001,
        'num_try': 1,           # Number of trials for averaging results
        'max_epochs': 100,

        # Photonic/Hardware Simulation Parameters
        'leverage': 0,          # Hardware leverage factor
        'kernel_size': 0,       # 0: No splitting
        
        # FFT Features (if applicable)
        'fft_dim': 32,
        'enc_out': 17,
        'compressed_dim': 17
    }

    # Parameters to include in folder names/logs
    folder_params = {k: params[k] for k in ['dataset', 'enc_type', 'cls_type']}

    # --- Preparation ---
    if save_results:
        save_experiment_report(variable_param, params, experiment_name=experiment_name)

    # Data Loading
    data_loaders = {
        'cifar-10': load_CIFAR10_data,
        'cinic-10': load_CINIC10_data,
        'mnist': load_MNIST_data,
        'fashion-mnist': load_Fmnist_data,
        'covtype': load_Covtype_data
    }
    
    if params['dataset'] not in data_loaders:
        raise ValueError(f"Dataset '{params['dataset']}' is not supported.")
        
    print(f"Loading dataset: {params['dataset']}...")
    data_train, data_test = data_loaders[params["dataset"]]()
    print("Data loaded successfully.")

    # --- Main Loop ---
    # Iterate over the variable parameter (e.g., different alpha values)
    # If variable_param is 'none', this loop runs once.
    
    results_summary = {
        'accuracies': [],
        'losses': [],
        'times': []
    }

    for variable_value in params[variable_param]:
        print(f'\n=== Running Experiment with {variable_param}: {variable_value} ===')
        
        current_var_metrics = {
            'last_loss': [],
            'loss_test_history': [],
            'process_time': [],
            'test_accuracy': []
        }

        # Multiple trials for statistical reliability
        for trial_idx in range(params['num_try']):
            print(f"  Trial {trial_idx + 1}/{params['num_try']}")
            
            # Prepare parameters for this specific run
            run_params = {k: v for k, v in params.items() if k not in ('none', variable_param)}
            
            # Update with current variable value and device
            run_params['device'] = device
            run_params['num_times'] = trial_idx
            if variable_param != 'none':
                run_params[variable_param] = variable_value

            # Execute Training
            loss_train, loss_test, pro_time, last_loss_test, test_acc, all_labels, all_preds = train_model(
                **run_params,
                data_train=data_train,
                data_test=data_test,
                ex_type=experiment_type
            )

            # Record metrics
            current_var_metrics['loss_test_history'].append(loss_test)
            current_var_metrics['process_time'].append(sum(pro_time))
            current_var_metrics['last_loss'].append(last_loss_test)
            current_var_metrics['test_accuracy'].append(test_acc)

            print(f"  -> Test Accuracy: {test_acc:.2f}")

            if save_results:
                trial_data = [loss_train, loss_test, all_labels, all_preds, test_acc]
                save_csv(trial_data, variable_param, variable_value, trial_idx, 
                         **folder_params, save_type='trial', experiment_name=experiment_name)

            if show_plots:
                plot_loss_curve(loss_train, loss_test, Save=save_results, Show=show_plots)
                plot_confusion_matrix(all_labels, all_preds, params["dataset"], test_acc, 
                                      Save=save_results, Show=show_plots)

        # --- Post-Trial Processing ---
        # Aggregate results for the current variable value
        mid_data = [
            current_var_metrics['loss_test_history'],
            current_var_metrics['test_accuracy'],
            current_var_metrics['last_loss'],
            current_var_metrics['process_time']
        ]

        if save_results:
            save_csv(mid_data, variable_param, variable_value, 0, 
                     **folder_params, save_type='mid', experiment_name=experiment_name)
        
        if show_plots:
            plot_errorbar_losscurve(current_var_metrics['loss_test_history'], Save=save_results, Show=show_plots)
            create_table(current_var_metrics['test_accuracy'], 
                         current_var_metrics['last_loss'], 
                         current_var_metrics['process_time'], 
                         Save=save_results, Show=show_plots)

        results_summary['accuracies'].append(current_var_metrics['test_accuracy'])
        results_summary['losses'].append(current_var_metrics['last_loss'])
        results_summary['times'].append(current_var_metrics['process_time'])

    # --- Finalization ---
    if variable_param != 'none' and save_results:
        final_data = [results_summary['accuracies'], results_summary['losses'], results_summary['times']]
        save_csv(final_data, variable_param, 0, 0, 
                 **folder_params, save_type='final', experiment_name=experiment_name)

    if save_results:
        create_result_pdf(variable_param, params, experiment_name=experiment_name)
        print(f"\nExperiment saved to: {experiment_name}")

if __name__ == "__main__":
    main()
