#!/usr/bin/env python3
"""
Simple Model Training Script for AI Health System
================================================

This script creates a simple model to demonstrate the system.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

from models import create_model, ModelTrainer

def create_synthetic_data(num_samples=1000, img_size=(224, 224)):
    """Create synthetic data for demonstration"""
    print("🎲 Creating synthetic training data...")
    
    # Create random images (simulating chest X-rays)
    X = np.random.random((num_samples, *img_size, 3)).astype(np.float32)
    
    # Create random labels (0: normal, 1: abnormal)
    y = np.random.randint(0, 2, (num_samples,))
    
    # Convert to one-hot encoding
    from tensorflow.keras.utils import to_categorical
    y = to_categorical(y, 2)
    
    # Split into train/validation
    split_idx = int(0.8 * num_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"✅ Synthetic data created:")
    print(f"  📊 Training: {len(X_train)} images")
    print(f"  📊 Validation: {len(X_val)} images")
    
    return (X_train, y_train), (X_val, y_val)

def main():
    """Main training function"""
    print("🚀 AI Health System - Simple Model Training")
    print("=" * 50)
    
    # Configuration
    model_type = 'custom'  # Use custom CNN for simplicity
    epochs = 10  # Fewer epochs for demo
    batch_size = 32
    learning_rate = 0.001
    
    print(f"⚙️  Configuration:")
    print(f"  Model: {model_type}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    
    # Create synthetic data
    (X_train, y_train), (X_val, y_val) = create_synthetic_data(num_samples=1000)
    
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
        train_data=X_train,
        train_labels=y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Evaluate model
    print(f"\n📊 Evaluating model...")
    test_metrics = trainer.evaluate(X_val, y_val)
    
    print(f"✅ Validation Results:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save model
    model_save_path = f'notebooks/saved_models/demo_{model_type.lower()}.h5'
    trainer.save_model(model_save_path)
    
    # Plot training history
    print(f"\n📈 Plotting training history...")
    os.makedirs('results', exist_ok=True)
    plot_save_path = f'results/training_history_{model_type.lower()}.png'
    
    # Create simple plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(plot_save_path)
    plt.close()
    
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
    print(f"2. 📊 Load your trained model: {model_save_path}")
    print(f"3. 🖼️  Upload X-ray images for analysis")
    
    print(f"\n💡 Note: This is a demo model with synthetic data.")
    print(f"   For real medical use, train on actual labeled chest X-ray data.")

if __name__ == "__main__":
    main()
