import torch
import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath("..")) 

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
print(f'Using device: {device}')

from dataloader.dataloader import load_MNIST_data,load_CIFAR10_data
from train.evaluate import show_images
from dataloader.dataloader import get_new_dataloader
#data---------------------------------------------
dataset = 'cifar-10'
batch_size = 64
data_train,data_test = load_CIFAR10_data()
rain_dataloader,test_dataloader = get_new_dataloader(data_train,
                                                            data_test,batch_size)
#-------------------------------------------------
fixed_indices = {1:1,3:1,5:1} #対象のラベルとインデックスを指定

for test_images, test_labels in test_dataloader:
    print(f"Test images shape: {test_images.shape}")
    show_images(test_images, test_labels,dataset,fixed_indices)  
    break  # Just to check the first batch
plt.show()