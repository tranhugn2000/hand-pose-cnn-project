import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical # type: ignore

train_df = pd.read_csv('sign_mnist_train.csv')
test_df = pd.read_csv('sign_mnist_test.csv')

train_labels = train_df['label'].values
train_images = train_df.drop('label', axis=1).values
test_labels = test_df['label'].values
test_images = test_df.drop('label', axis=1).values

train_images = train_images / 255.0
test_images = test_images / 255.0
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

num_classes = 25
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

print("Dữ liệu đã sẵn sàng: Train shape:", X_train.shape, "Val shape:", X_val.shape, "Test shape:", test_images.shape)