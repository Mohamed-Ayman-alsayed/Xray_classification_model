from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Build cross-platform paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "unsupervised_2.0.h5")
SAMPLE_IMAGE_PATH = os.path.join(
    os.path.dirname(BASE_DIR),
    "notebooks",
    "data",
    "processed",
    "0ae738d0-5bdd-4ebf-9c1f-6b10a25be92a.png",
)

# Load your trained model
model = load_model(MODEL_PATH)

# Load a single known X-ray
img = image.load_img(SAMPLE_IMAGE_PATH, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x /= 255.0  # normalize like during training

# Predict
pred = model.predict(x)
print("Colab model prediction:", pred)