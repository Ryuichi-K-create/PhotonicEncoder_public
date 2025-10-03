import torch
import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
print(f'Using device: {device}')

from dataloader.dataloader import load_MNIST_data,load_CIFAR10_data, load_CINIC10_data, load_Fmnist_data
from train.evaluate import show_images
from dataloader.dataloader import get_new_dataloader
#data---------------------------------------------
dataset ='fashion-mnist' # 'cifar10', 'cinic10', 'mnist'
fixed_indices = {1:1,4:3,5:2}

#-------------------------------------------------
load_data = {
    'mnist': load_MNIST_data,
    'cifar-10': load_CIFAR10_data,
    'cinic-10': load_CINIC10_data,
    'fashion-mnist': load_Fmnist_data
}

batch_size = 64
data_train,data_test = load_data[dataset]()
train_dataloader,test_dataloader = get_new_dataloader(data_train,
                                                            data_test,batch_size)
#-------------------------------------------------

for test_images, test_labels in test_dataloader:
    show_images(test_images, test_labels,dataset,fixed_indices)  
    break  # Just to check the first batch
plt.show()
