#!/usr/bin/env python3
"""
Create Dummy Model for Testing
==============================

This script creates a simple dummy model that can be loaded in the GUI.
"""

import os
import sys
import numpy as np

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

from models import create_model, ModelTrainer

def main():
    """Create a dummy model"""
    print("🎲 Creating dummy model for testing...")
    
    # Create a simple custom model
    model = create_model(
        model_type='custom',
        input_shape=(224, 224, 3),
        num_classes=2
    )
    
    # Create trainer
    trainer = ModelTrainer(model)
    
    # Create some dummy weights (random initialization)
    print("⚙️  Initializing model with random weights...")
    
    # Save the model
    model_save_path = 'notebooks/saved_models/dummy_model.h5'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    trainer.save_model(model_save_path)
    
    print(f"✅ Dummy model created and saved to: {model_save_path}")
    print(f"📊 Model summary:")
    model.summary()
    
    print(f"\n🚀 You can now:")
    print(f"1. 🖥️  Launch GUI: python run_gui.py")
    print(f"2. 📊 Load this model: {model_save_path}")
    print(f"3. 🖼️  Upload X-ray images for testing")
    
    print(f"\n💡 Note: This is a dummy model with random weights.")
    print(f"   It will give random predictions - use only for testing the GUI interface.")

if __name__ == "__main__":
    main()
