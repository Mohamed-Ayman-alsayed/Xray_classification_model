
#!/usr/bin/env python3
"""
GUI Prediction Script for Unsupervised Model
============================================

Tkinter-based interface to load a chest X-ray, run it through the
trained autoencoder (unsupervised_2.0.h5), and display anomaly results.
"""

import os
import numpy as np
import cv2
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow import keras
import tkinter as tk
from tkinter import filedialog, messagebox

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
        return np.expand_dims(img_array, axis=0), img
    except Exception as e:
        messagebox.showerror("Error", f"❌ Failed to load {image_path}: {e}")
        return None, None

# ========================
# 3. Prediction Function
# ========================
def predict_anomaly(model, image, threshold=0.01):
    """Reconstruct image and decide anomaly."""
    reconstruction = model.predict(image, verbose=0)
    mse = np.mean(np.square(image - reconstruction))

    if mse > threshold:
        result = f"🚨 Anomaly detected (MSE={mse:.6f})"
    else:
        result = f"✅ Normal (MSE={mse:.6f})"

    return result

# ========================
# 4. GUI App
# ========================
class AnomalyDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chest X-Ray Anomaly Detector")
        self.root.geometry("500x500")

        # Buttons
        self.browse_btn = tk.Button(root, text="📂 Browse Image", command=self.load_image)
        self.browse_btn.pack(pady=10)

        # Image Display
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        # Result Display
        self.result_label = tk.Label(root, text="Result will appear here", font=("Arial", 14))
        self.result_label.pack(pady=20)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an X-ray",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.dcm")]
        )
        if not file_path:
            return

        image_array, pil_image = preprocess_image(file_path)
        if image_array is None:
            return

        # Show image preview
        img_preview = pil_image.resize((250, 250))
        img_tk = ImageTk.PhotoImage(img_preview)
        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk

        # Predict anomaly
        result = predict_anomaly(model, image_array, threshold=0.01)
        self.result_label.config(text=result)

# ========================
# 5. Run App
# ========================
if __name__ == "__main__":
    root = tk.Tk()
    app = AnomalyDetectorApp(root)
    root.mainloop()
