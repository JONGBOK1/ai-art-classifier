# preprocessing/data_split.py

import os
import shutil
import random
from pathlib import Path

SRC_ROOT = 'dataset'
DST_ROOT = 'dataset_split'
split_ratio = {'train': 0.7, 'val': 0.15, 'test': 0.15}
classes = ['real', 'ai']

for cls in classes:
    files = list(Path(os.path.join(SRC_ROOT, cls)).glob("*.jpg"))  # png일 경우 확장자 추가
    random.shuffle(files)

    train_end = int(len(files) * split_ratio['train'])
    val_end = train_end + int(len(files) * split_ratio['val'])

    subsets = {
        'train': files[:train_end],
        'val': files[train_end:val_end],
        'test': files[val_end:]
    }

    for subset, subset_files in subsets.items():
        save_dir = os.path.join(DST_ROOT, subset, cls)
        os.makedirs(save_dir, exist_ok=True)

        for file_path in subset_files:
            shutil.copy(file_path, save_dir)

print("✅ 데이터 분할 완료!")
