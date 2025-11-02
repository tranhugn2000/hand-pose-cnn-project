# type: ignore
# train.py
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

os.makedirs("models", exist_ok=True)

# 1) Load dữ liệu
X_train = np.load('data/X_train.npy')
X_val   = np.load('data/X_val.npy')
y_train = np.load('data/y_train.npy')
y_val   = np.load('data/y_val.npy')

# 2) Load model chưa huấn luyện
model = load_model('models/untrained_model.keras')

# 3) Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(X_train)

# 4) Callbacks
ckpt = ModelCheckpoint('models/best_model.keras', monitor='val_accuracy',
                       save_best_only=True, mode='max', verbose=1)
es   = EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True, verbose=1)
rlr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,
                         min_lr=1e-5, verbose=1)

# 5) Train
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64, shuffle=True),
    epochs=40,
    validation_data=(X_val, y_val),
    callbacks=[ckpt, es, rlr],
    verbose=1
)

# 6) Lưu thêm định dạng khác
best_model = load_model('models/best_model.keras')

print("✅ Đã lưu mô hình: models/best_model.keras")
