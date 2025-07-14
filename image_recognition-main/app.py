"""
Flask + TensorFlow API (Render‑ / Railway‑ready)
Lazy‑loads model so the container boots fast.
"""

import os
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# ───────── CONFIG ─────────
BASE_DIR   = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "checkpoints" / "mobilenetv3_enhanced" / "mobilenetv3_enhanced.keras"
LABELS_PATH= BASE_DIR / "checkpoints" / "mobilenetv3_enhanced" / "labels.txt"
UPLOADS_DIR= BASE_DIR / "uploads"
IMG_SIZE   = (224, 224)

os.makedirs(UPLOADS_DIR, exist_ok=True)

# ───────── VERSION SAFETY ─────────
# If someone installs NumPy 2 by accident, bail early with a clear msg
import numpy
if numpy.__version__.startswith("2."):
    raise RuntimeError(
        f"🚨 NumPy {numpy.__version__} is unsupported by TensorFlow. "
        "Pin numpy<2 in requirements.txt")

# Mixed‑precision only if a GPU is available
if tf.config.list_physical_devices('GPU'):
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
else:
    tf.keras.mixed_precision.set_global_policy("float32")

# ───────── APP ─────────
app = Flask(__name__)
CORS(app)

_model = None
CLASS_NAMES = []

def load_model_once():
    global _model, CLASS_NAMES
    if _model is None:
        print("🔄  Loading TensorFlow model …")
        _model = tf.keras.models.load_model(MODEL_PATH)
        CLASS_NAMES = (LABELS_PATH.read_text().splitlines()
                       if LABELS_PATH.exists() else
                       [str(i) for i in range(_model.output_shape[-1])])
        print("✅  Model ready:", CLASS_NAMES)

@app.route("/", methods=["GET"])
def health():
    return {"status": "ok"}, 200

@app.route("/predict", methods=["POST"])
def predict():
    load_model_once()

    file = request.files.get("file")
    if not file:
        return jsonify(error="No file uploaded"), 400

    tmp = UPLOADS_DIR / file.filename
    file.save(tmp)

    try:
        img = image.load_img(tmp, target_size=IMG_SIZE)
        arr = image.img_to_array(img)
        arr = preprocess_input(arr)
        preds = _model.predict(np.expand_dims(arr, 0))[0]

        idx  = int(np.argmax(preds))
        conf = float(preds[idx]) * 100
        return {"class": CLASS_NAMES[idx], "confidence": f"{conf:.2f}"}
    finally:
        tmp.unlink(missing_ok=True)

# ───────── Local dev (python app.py) ─────────
if __name__ == "__main__":
    app.run(debug=True, port=8080)
