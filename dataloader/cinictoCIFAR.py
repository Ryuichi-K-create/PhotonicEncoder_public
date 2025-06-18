import os
import shutil
import sys

cinic_dir = '/Users/konishi/PhotnicEncoder/dataloader/samples/cinic10_data/valid'  # train, valid, test どれでも
output_dir = '/Users/konishi/PhotnicEncoder/dataloader/samples/cinic10_data/valid_cifar10'

os.makedirs(output_dir, exist_ok=True)

for class_name in os.listdir(cinic_dir):
    class_dir = os.path.join(cinic_dir, class_name)
    if not os.path.isdir(class_dir):
        continue
    out_class_dir = os.path.join(output_dir, class_name)
    os.makedirs(out_class_dir, exist_ok=True)
    for fname in os.listdir(class_dir):
        if 'cifar10' in fname:
            src = os.path.join(class_dir, fname)
            dst = os.path.join(out_class_dir, fname)
            shutil.copy(src, dst)

        sys.stderr.write(f'\rProcessing {class_name} class: {len(os.listdir(out_class_dir))} files copied')
        sys.stderr.flush()
    