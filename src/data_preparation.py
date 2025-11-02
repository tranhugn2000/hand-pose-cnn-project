# data_preparation.py
import os, json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 1) Load CSV Kaggle
train_df = pd.read_csv('data/sign_mnist_train.csv')
test_df  = pd.read_csv('data/sign_mnist_test.csv')

# 2) Tách nhãn & ảnh
y_train_raw = train_df['label'].values
X_train_raw = train_df.drop('label', axis=1).values

y_test_raw  = test_df['label'].values
X_test_raw  = test_df.drop('label', axis=1).values

# 3) Remap label: nén khoảng trống ở index 9 (J) => 24 lớp 0..23
def compress_label(v):
    return v if v < 9 else v - 1  # 10->9, 11->10, ..., 24->23

y_train_int = np.array([compress_label(v) for v in y_train_raw], dtype=np.int64)
y_test_int  = np.array([compress_label(v) for v in y_test_raw],  dtype=np.int64)

# 4) Chuẩn hóa ảnh & reshape
X_train = (X_train_raw / 255.0).reshape(-1, 28, 28, 1).astype("float32")
X_test  = (X_test_raw  / 255.0).reshape(-1, 28, 28, 1).astype("float32")

# 5) Chia val có stratify theo nhãn số
X_train, X_val, y_train_int, y_val_int = train_test_split(
    X_train, y_train_int, test_size=0.2, random_state=42, stratify=y_train_int
)

# 6) One-hot
num_classes = 24
y_train = to_categorical(y_train_int, num_classes)
y_val   = to_categorical(y_val_int,   num_classes)
y_test  = to_categorical(y_test_int,  num_classes)

print("Dữ liệu đã sẵn sàng:")
print("  Train:", X_train.shape, y_train.shape)
print("  Val  :", X_val.shape,   y_val.shape)
print("  Test :", X_test.shape,  y_test.shape)

# 7) Lưu .npy
np.save('data/X_train.npy', X_train)
np.save('data/X_val.npy',   X_val)
np.save('data/y_train.npy', y_train)
np.save('data/y_val.npy',   y_val)
np.save('data/X_test.npy',  X_test)
np.save('data/y_test.npy',  y_test)

# 8) Lưu mapping nhãn 24 lớp: A..Y (bỏ J)
LABELS_24 = ["A","B","C","D","E","F","G","H","I","K","L","M",
             "N","O","P","Q","R","S","T","U","V","W","X","Y"]
with open("models/labels_24.json", "w", encoding="utf-8") as f:
    json.dump(LABELS_24, f, ensure_ascii=False, indent=2)
print("Đã lưu labels vào models/labels_24.json")
