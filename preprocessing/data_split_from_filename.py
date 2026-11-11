import os
import shutil
from pathlib import Path
import random

SRC_TRAIN = 'Real_AI_SD_LD_Dataset/train'
SRC_TEST = 'Real_AI_SD_LD_Dataset/test'
DST_ROOT = 'dataset_split'
VAL_RATIO = 0.2

def classify_by_filename(files):
    real_files = [f for f in files if "AI" not in f.name]
    ai_files = [f for f in files if "AI" in f.name]
    return real_files, ai_files

def split_and_copy(files, cls_name, subset):
    random.shuffle(files)
    val_count = int(len(files) * VAL_RATIO)

    if subset == 'train':
        train_files = files[val_count:]
        val_files = files[:val_count]

        for f in train_files:
            save_to = os.path.join(DST_ROOT, 'train', cls_name)
            os.makedirs(save_to, exist_ok=True)
            shutil.copy(f, save_to)

        for f in val_files:
            save_to = os.path.join(DST_ROOT, 'val', cls_name)
            os.makedirs(save_to, exist_ok=True)
            shutil.copy(f, save_to)

    elif subset == 'test':
        for f in files:
            save_to = os.path.join(DST_ROOT, 'test', cls_name)
            os.makedirs(save_to, exist_ok=True)
            shutil.copy(f, save_to)

# ▶ train + val
train_files = list(Path(SRC_TRAIN).glob("*.jpg"))
real_train, ai_train = classify_by_filename(train_files)
split_and_copy(real_train, 'real', 'train')
split_and_copy(ai_train, 'ai', 'train')

# ▶ test
test_files = list(Path(SRC_TEST).glob("*.jpg"))
real_test, ai_test = classify_by_filename(test_files)
split_and_copy(real_test, 'real', 'test')
split_and_copy(ai_test, 'ai', 'test')

print("✅ 파일 이름 기반 데이터 분할 완료!")
