"""
Configuration file for AI Health System
"""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'notebooks', 'data')  # Updated to notebooks/data
MODELS_DIR = os.path.join(BASE_DIR, 'notebooks', 'saved_models')  # Updated to notebooks/saved_models
REPORTS_DIR = os.path.join(DATA_DIR, 'reports')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Data processing settings
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 0.001

# Model settings
SUPPORTED_MODELS = ['custom', 'ResNet50', 'VGG16', 'EfficientNetB0', 'DenseNet121']
DEFAULT_MODEL_TYPE = 'custom'
DEFAULT_NUM_CLASSES = 2

# Data augmentation settings
AUGMENTATION_PROBABILITY = 0.5
ROTATION_RANGE = (-15, 15)
BRIGHTNESS_RANGE = (0.8, 1.2)
CONTRAST_RANGE = (0.8, 1.2)

# Report settings
DEFAULT_REPORT_FORMAT = 'pdf'
REPORT_TEMPLATE_DIR = os.path.join(BASE_DIR, 'src', 'templates')

# GUI settings
GUI_WINDOW_SIZE = (1200, 800)
GUI_MIN_IMAGE_SIZE = (400, 400)

# Validation settings
DEFAULT_VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# File extensions
SUPPORTED_IMAGE_FORMATS = ['.png', '.jpg', '.jpeg', '.dcm']
SUPPORTED_MODEL_FORMATS = ['.h5', '.hdf5']

# Performance settings
MAX_WORKERS = 4
CHUNK_SIZE = 1000

# Logging settings
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)
