"""
AI Health System - Medical Image Analysis Package
================================================

A comprehensive system for chest X-ray analysis using deep learning.
Features include data preprocessing, CNN models, and automated report generation.
"""

__version__ = "1.0.0"
__author__ = "AI Health System Team"
__email__ = "team@aihealthsystem.com"

# Conditional imports to avoid errors during development
try:
    from . import data_processing
    from . import models
    from . import utils
    from . import gui
    from . import reporting
    
    __all__ = [
        'data_processing',
        'models', 
        'utils',
        'gui',
        'reporting'
    ]
except ImportError:
    # During development or if modules aren't fully set up yet
    __all__ = []
    pass
