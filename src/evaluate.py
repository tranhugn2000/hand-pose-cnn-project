import numpy as np
from tensorflow.keras.models import load_model

X_test = np.load('data/X_test.npy')   # đã /255, reshape (N,28,28,1)
y_test = np.load('data/y_test.npy')   # one-hot 24

model = load_model('models/best_model.keras')
print("[INFO] model.output_shape:", model.output_shape)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"[TEST] accuracy = {acc:.4f}, loss = {loss:.4f}")