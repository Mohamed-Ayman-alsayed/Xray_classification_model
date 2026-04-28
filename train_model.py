#!/usr/bin/env python3
"""
Model Training Script for AI Health System
=========================================

This script loads the processed dataset and trains a model on your chest X-ray data.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

from models import create_model, ModelTrainer
from data_processing import ChestXRayProcessor
from utils import plot_training_history, create_directory_structure

def load_processed_data(data_dir: str):
    """Load processed images and create labels"""
    print("📁 Loading processed data...")
    
    train_dir = os.path.join(data_dir, 'processed', 'train')
    test_dir = os.path.join(data_dir, 'processed', 'test')
    val_dir = os.path.join(data_dir, 'processed', 'validation')
    
    # Check if validation split exists, if not create it
    if not os.path.exists(val_dir) or len(os.listdir(val_dir)) == 0:
        print("🔄 Creating train/validation split...")
        processor = ChestXRayProcessor(data_dir)
        processor.create_train_val_split(train_dir)
    
    # Load training data
    train_images = []
    train_labels = []
    
    print("📊 Loading training images...")
    for class_name in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_name)
        if os.path.isdir(class_path):
            class_id = 0 if class_name.lower() == 'normal' else 1
            print(f"  📁 {class_name}: class_id={class_id}")
            
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith('.png'):
                    img_path = os.path.join(class_path, img_file)
                    train_images.append(img_path)
                    train_labels.append(class_id)
    
    # Load validation data
    val_images = []
    val_labels = []
    
    print("📊 Loading validation images...")
    for class_name in os.listdir(val_dir):
        class_path = os.path.join(val_dir, class_name)
        if os.path.isdir(class_path):
            class_id = 0 if class_name.lower() == 'normal' else 1
            
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith('.png'):
                    img_path = os.path.join(class_path, img_file)
                    val_images.append(img_path)
                    val_labels.append(class_id)
    
    # Load test data
    test_images = []
    test_labels = []
    
    print("📊 Loading test images...")
    for class_name in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_name)
        if os.path.isdir(class_path):
            class_id = 0 if class_name.lower() == 'normal' else 1
            
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith('.png'):
                    img_path = os.path.join(class_path, img_file)
                    test_images.append(img_path)
                    test_labels.append(class_id)
    
    print(f"✅ Data loaded:")
    print(f"  📊 Training: {len(train_images)} images")
    print(f"  📊 Validation: {len(val_images)} images")
    print(f"  📊 Test: {len(test_images)} images")
    
    return {
        'train': (train_images, train_labels),
        'validation': (val_images, val_labels),
        'test': (test_images, test_labels)
    }

def preprocess_images(image_paths, target_size=(224, 224)):
    """Preprocess images for training"""
    print("🔄 Preprocessing images...")
    
    images = []
    for i, img_path in enumerate(image_paths):
        if i % 1000 == 0:
            print(f"  📸 Processed {i}/{len(image_paths)} images...")
        
        try:
            # Load image
            img = plt.imread(img_path)
            
            # Convert to RGB if grayscale
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[2] == 4:  # RGBA
                img = img[:, :, :3]
            
            # Resize
            img = plt.imresize(img, target_size)
            
            # Normalize to [0, 1]
            if img.max() > 1:
                img = img.astype(np.float32) / 255.0
            
            images.append(img)
            
        except Exception as e:
            print(f"⚠️  Error loading {img_path}: {e}")
            # Add a placeholder image
            placeholder = np.zeros((*target_size, 3), dtype=np.float32)
            images.append(placeholder)
    
    return np.array(images)

def create_labels(labels, num_classes=2):
    """Convert labels to one-hot encoding"""
    from sklearn.preprocessing import to_categorical
    return to_categorical(labels, num_classes)

def main():
    """Main training function"""
    print("🚀 AI Health System - Model Training")
    print("=" * 50)
    
    # Configuration
    data_dir = './notebooks/data'
    model_type = 'ResNet50'  # or 'custom', 'VGG16', 'EfficientNetB0'
    epochs = 50
    batch_size = 32
    learning_rate = 0.001
    
    print(f"⚙️  Configuration:")
    print(f"  Model: {model_type}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Data directory: {data_dir}")
    
    # Check if data exists
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    # Load data
    data = load_processed_data(data_dir)
    
    if len(data['train'][0]) == 0:
        print("❌ No training data found!")
        print("💡 Make sure to run data processing first:")
        print("   ai-health process --data-dir ./notebooks/data")
        return
    
    # Preprocess images
    print("\n🔄 Preprocessing training images...")
    train_images = preprocess_images(data['train'][0])
    train_labels = create_labels(data['train'][1])
    
    print("🔄 Preprocessing validation images...")
    val_images = preprocess_images(data['validation'][0])
    val_labels = create_labels(data['validation'][1])
    
    print("🔄 Preprocessing test images...")
    test_images = preprocess_images(data['test'][0])
    test_labels = create_labels(data['test'][1])
    
    print(f"✅ Preprocessing complete:")
    print(f"  📊 Training images: {train_images.shape}")
    print(f"  📊 Validation images: {val_images.shape}")
    print(f"  📊 Test images: {test_images.shape}")
    
    # Create model
    print(f"\n🎯 Creating {model_type} model...")
    model = create_model(
        model_type=model_type,
        input_shape=(224, 224, 3),
        num_classes=2
    )
    
    # Create trainer
    trainer = ModelTrainer(model)
    
    # Train model
    print(f"\n🚀 Starting training...")
    history = trainer.train(
        train_data=train_images,
        train_labels=train_labels,
        validation_data=(val_images, val_labels),
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Evaluate model
    print(f"\n📊 Evaluating model...")
    test_metrics = trainer.evaluate(test_images, test_labels)
    
    print(f"✅ Test Results:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save model
    model_save_path = f'notebooks/saved_models/best_{model_type.lower()}.h5'
    trainer.save_model(model_save_path)
    
    # Plot training history
    print(f"\n📈 Plotting training history...")
    plot_save_path = f'results/training_history_{model_type.lower()}.png'
    os.makedirs('results', exist_ok=True)
    plot_training_history(history, save_path=plot_save_path)
    
    # Save results
    results = {
        'model_type': model_type,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'test_metrics': test_metrics,
        'model_path': model_save_path,
        'training_history': {
            'loss': [float(x) for x in history.history['loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_loss': [float(x) for x in history.history.get('val_loss', [])],
            'val_accuracy': [float(x) for x in history.history.get('val_accuracy', [])]
        }
    }
    
    results_save_path = f'results/training_results_{model_type.lower()}.json'
    import json
    with open(results_save_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n🎉 Training complete!")
    print(f"✅ Model saved to: {model_save_path}")
    print(f"✅ Results saved to: {results_save_path}")
    print(f"✅ Training plots saved to: {plot_save_path}")
    
    print(f"\n🚀 Next steps:")
    print(f"1. 🖥️  Launch GUI: python run_gui.py")
    print(f"2. 📊 Load your trained model in the GUI")
    print(f"3. 🖼️  Upload X-ray images for analysis")

if __name__ == "__main__":
    main()
