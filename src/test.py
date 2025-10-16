import os, time, numpy as np, pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ====== config ======
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TRAIN_CSV = os.path.join(DATA_DIR, "sign_mnist_train.csv")
TEST_CSV  = os.path.join(DATA_DIR, "sign_mnist_test.csv")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

IDX2CHAR = {
    0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',
    6:'G',7:'H',8:'I',9:'K',10:'L',11:'M',
    12:'N',13:'O',14:'P',15:'Q',16:'R',17:'S',
    18:'T',19:'U',20:'V',21:'W',22:'X',23:'Y'
}
NUM_CLASSES = len(IDX2CHAR)

def load_csv(path):
    df = pd.read_csv(path)
    y = df['label'].values.astype('int32')
    X = df.drop(columns=['label']).values.astype('float32')
    X = (X / 255.0).reshape((-1, 28, 28, 1))
    return X, y

def make_datasets(batch_size=128, val_split=0.1, shuffle=10000):
    X_train, y_train = load_csv(TRAIN_CSV)
    X_test,  y_test  = load_csv(TEST_CSV)

    n = len(X_train)
    idx = np.random.RandomState(42).permutation(n)
    split = int(n * (1 - val_split))
    tr_idx, va_idx = idx[:split], idx[split:]

    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_va, y_va = X_train[va_idx], y_train[va_idx]

    aug = keras.Sequential([
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.05),
        layers.RandomTranslation(0.05, 0.05),
    ])

    def train_map(x, y):
        x = tf.cast(x, tf.float32)
        x = aug(x, training=True)
        return x, y

    ds_tr = tf.data.Dataset.from_tensor_slices((X_tr, y_tr)) \
        .shuffle(shuffle).map(train_map, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(batch_size).prefetch(tf.data.AUTOTUNE)

    ds_va = tf.data.Dataset.from_tensor_slices((X_va, y_va)) \
        .batch(batch_size).prefetch(tf.data.AUTOTUNE)

    ds_te = tf.data.Dataset.from_tensor_slices((X_test, y_test)) \
        .batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds_tr, ds_va, ds_te, (X_test, y_test)

def build_model():
    inputs = layers.Input(shape=(28,28,1))
    x = layers.Conv2D(32, (3,3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3,3), activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3,3), activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3,3), activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="sign_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def plot_history(hist, outpath):
    plt.figure()
    plt.plot(hist.history["accuracy"], label="train_acc")
    plt.plot(hist.history["val_accuracy"], label="val_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(outpath); plt.close()

def main():
    print("Loading data...")
    ds_tr, ds_va, ds_te, (X_test, y_test) = make_datasets()

    print("Building model...")
    model = build_model()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=7, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(OUT_DIR, "model_best.h5"),
            monitor="val_accuracy", save_best_only=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
        )
    ]

    print("Training...")
    history = model.fit(
        ds_tr,
        validation_data=ds_va,
        epochs=40,
        verbose=2,
        callbacks=callbacks
    )

    model_path = os.path.join(OUT_DIR, "model.h5")
    model.save(model_path)
    print(f"Saved model to {model_path}")

    # plot learning curve
    plot_path = os.path.join(OUT_DIR, "learning_curve.png")
    plot_history(history, plot_path)
    print(f"Saved plot to {plot_path}")

    # Evaluate
    print("Evaluating...")
    test_loss, test_acc = model.evaluate(ds_te, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    # Detailed metrics
    y_pred = np.argmax(model.predict(X_test, batch_size=256, verbose=0), axis=1)
    print(classification_report(y_test, y_pred, digits=4))

    # Confusion matrix figure
    cm = confusion_matrix(y_test, y_pred)
    fig = plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.colorbar()
    plt.tight_layout()
    cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path); plt.close(fig)
    print(f"Saved confusion matrix to {cm_path}")

if __name__ == "__main__":
    main()
