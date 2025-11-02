# app.py
import os, json
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, flash
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = "change-me"

ALLOWED = {"png", "jpg", "jpeg"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED

# ----- Load model (24 lớp) -----
def load_best_model():
    candidates = [
        os.path.join("models", "best_model.keras"),
        "best_model.keras"
    ]
    for p in candidates:
        if os.path.exists(p):
            print(f"[INFO] Loading model: {p}")
            return load_model(p)
    raise FileNotFoundError("Không tìm thấy models/best_model.keras")

model = load_best_model()

# Xác nhận 24 class
out_classes = int(model.output_shape[-1])
if out_classes != 24:
    raise ValueError(f"Model có {out_classes} class, app yêu cầu 24 class (A..Y, bỏ J).")

# ----- Load labels -----
labels_path = os.path.join("models", "labels_24.json")
if not os.path.exists(labels_path):
    # fallback an toàn (khớp SignMNIST)
    LABELS = ["A","B","C","D","E","F","G","H","I","K","L","M",
              "N","O","P","Q","R","S","T","U","V","W","X","Y"]
else:
    with open(labels_path, "r", encoding="utf-8") as f:
        LABELS = json.load(f)
if len(LABELS) != 24:
    raise ValueError("labels_24.json không hợp lệ. Cần 24 nhãn.")

# ----- Preprocess giống MNIST -----
def preprocess_bytes_to_tensor(b: bytes):
    import numpy as np, cv2
    arr = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Ảnh không hợp lệ hoặc định dạng không hỗ trợ.")
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    x = img.astype("float32") / 255.0
    x = x.reshape(1, 28, 28, 1)
    return x # trả thêm ảnh trung gian để debug/hiển thị optional

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            flash("Không có file ảnh.")
            return redirect(request.url)
        f = request.files["image"]
        if f.filename == "":
            flash("Chưa chọn file.")
            return redirect(request.url)
        if not allowed_file(f.filename):
            flash("Định dạng không hỗ trợ. Hãy chọn PNG/JPG.")
            return redirect(request.url)

        try:
            X = preprocess_bytes_to_tensor(f.read())
            preds = model.predict(X, verbose=0)[0]  # (24,)

            # Top-5
            topk_idx = preds.argsort()[-5:][::-1]
            best_idx = int(topk_idx[0])
            best_prob = float(preds[best_idx])
            top5 = [(LABELS[int(i)], float(preds[int(i)])) for i in topk_idx]

            return render_template(
                "result.html",
                letter=LABELS[best_idx],
                prob=best_prob,
                top5=top5
            )
        except Exception as e:
            flash(f"Lỗi xử lý/dự đoán: {e}")
            return redirect(request.url)

    return render_template("index.html")

if __name__ == "__main__":
    # Chạy server dev
    app.run(host="0.0.0.0", port=5000, debug=True)
