# type: ignore
from tensorflow.keras.models import load_model 

import numpy as np

model = load_model('best_model.keras')

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test Accuracy: {test_acc * 100:.2f}%')

sample_image = test_images[0].reshape(1, 28, 28, 1)
prediction = model.predict(sample_image)
predicted_label = np.argmax(prediction)
true_label = np.argmax(test_labels[0])
print(f'Dự đoán: {chr(65 + predicted_label)} (True: {chr(65 + true_label)})')