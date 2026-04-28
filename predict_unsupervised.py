#!/usr/bin/env python3
"""
Prediction Script for Unsupervised Model
========================================

Loads the trained autoencoder (unsupervised_2.0.h5) and tests chest X-rays
for anomalies based on reconstruction error.
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# ========================
# 1. Load Model
# ========================
# Build a cross-platform path to the model file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "unsupervised_2.0.h5")

print(f"📂 Loading model from: {MODEL_PATH}")
model = keras.models.load_model(MODEL_PATH, compile=False)

# ========================
# 2. Preprocess Image
# ========================
def preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess image (jpg, png, jpeg, dcm)."""
    try:
        if image_path.lower().endswith(".dcm"):
            import pydicom
            ds = pydicom.dcmread(image_path)
            img_array = ds.pixel_array
            img_array = cv2.resize(img_array, target_size)
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        else:
            img = Image.open(image_path).convert("RGB")
            img = img.resize(target_size)
            img_array = np.array(img)
        img_array = img_array.astype(np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)  # add batch dimension
    except Exception as e:
        print(f"❌ Failed to load {image_path}: {e}")
        sys.exit(1)

# ========================
# 3. Prediction Function
# ========================
def predict_anomaly(model, image, threshold=0.01):
    """Reconstruct image and decide anomaly."""
    reconstruction = model.predict(image, verbose=0)
    mse = np.mean(np.square(image - reconstruction))
    print(f"📊 Reconstruction error (MSE): {mse:.6f}")

    if mse > threshold:
        print("🚨 Potential anomaly detected")
    else:
        print("✅ Likely normal")

    return mse

# ========================
# 4. Main
# ========================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_unsupervised.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    image = preprocess_image(image_path)
    mse = predict_anomaly(model, image, threshold=0.01)
