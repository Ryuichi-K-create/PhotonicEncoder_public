import os
import numpy as np
from PIL import Image
import sys

data_root = '/Users/konishi/PhotnicEncoder/dataloader/samples/cinic10_data'
split = 'train'  # 'train', 'valid', 'test' で切り替え
data_dir = os.path.join(data_root, split)

all_imgs = []
all_labels = []
class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

for label, class_name in enumerate(class_names):
    class_dir = os.path.join(data_dir, class_name)
    for fname in os.listdir(class_dir):
        fpath = os.path.join(class_dir, fname)
        if not os.path.isfile(fpath):
            continue  # ファイルでなければスキップ
        try:
            img = Image.open(fpath).convert('RGB')
            all_imgs.append(np.array(img))
            all_labels.append(label)
        except Exception as e:
            print(f"Skipping {fpath}: {e}")
        
        sys.stderr.write(f'\rProcessing {split} data: {len(all_imgs)} images loaded')
        sys.stderr.flush()
       
if len(all_imgs) == 0:
    raise RuntimeError("画像が1枚も読み込まれていません。パスやディレクトリ構成を再確認してください。")

all_imgs = np.stack(all_imgs)
all_labels = np.array(all_labels)

save_dir = '/Users/konishi/PhotnicEncoder/dataloader/samples/cinic10_data/train_main'
os.makedirs(save_dir, exist_ok=True)
np.save(os.path.join(save_dir, f'cinic10_{split}_imgs.npy'), all_imgs)
np.save(os.path.join(save_dir, f'cinic10_{split}_labels.npy'), all_labels)