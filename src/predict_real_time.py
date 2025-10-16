# type: ignore
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Tải mô hình đã huấn luyện
model = load_model('models/best_model.keras')

# Hàm xử lý ảnh thực tế
def preprocess_image(image_path):
    # Đọc ảnh và chuyển thành grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Không thể đọc ảnh từ đường dẫn: " + image_path)
    
    # Resize về 28x28
    img = cv2.resize(img, (28, 28))
    
    # Chuẩn hóa giá trị pixel (0-255 -> 0-1)
    img = img / 255.0
    
    # Reshape thành (1, 28, 28, 1) để dự đoán
    img = img.reshape(1, 28, 28, 1)
    return img

# Hàm dự đoán
def predict_image(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)
    return chr(65 + predicted_label)  # Chuyển số thành chữ cái (A-Y)

# Thử nghiệm với ảnh thực tế
if __name__ == "__main__":
    # Đường dẫn tới ảnh thực tế (thay bằng ảnh của bạn)
    image_path = "data/real_data/sample_hand.jpg"
    try:
        predicted_letter = predict_image(image_path)
        print(f"Dự đoán: {predicted_letter}")
        
        # Hiển thị ảnh đã xử lý (tùy chọn, để kiểm tra)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        cv2.imshow("Processed Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Lỗi: {e}")