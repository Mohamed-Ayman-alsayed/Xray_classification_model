#!/usr/bin/env python3
"""
Better Training for Higher Accuracy
==================================

This script implements key techniques to achieve >60% accuracy.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add current directory to path (since we're already in src)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from models import create_model

# Synthetic data function removed - using real dataset with ImageDataGenerator

def main():
    """Main training function"""
    print("🚀 AI Health System - Better Training for Higher Accuracy")
    print("=" * 55)
    
    # Configuration
    epochs = 80
    batch_size = 32
    learning_rate = 0.0005
    
    print(f"⚙️  Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    
    # Setup data generators
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.15
    )

    train_generator = train_datagen.flow_from_directory(
        "../notebooks/data/organized/train", 
        target_size=(224,224), 
        batch_size=batch_size,
        class_mode='categorical', 
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        "../notebooks/data/organized/train", 
        target_size=(224,224), 
        batch_size=batch_size,
        class_mode='categorical', 
        subset='validation'
    )

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        "../notebooks/data/organized/test", 
        target_size=(224,224), 
        batch_size=batch_size,
        class_mode='categorical', 
        shuffle=False
    )
    
    # Create model (use ResNet50 for better performance)
    print(f"\n🎯 Creating ResNet50 model...")
    model = create_model(
        model_type='ResNet50',
        input_shape=(224, 224, 3),
        num_classes=2
    )
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    print(f"\n🚀 Starting training...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        verbose=1
    )
    
    # Evaluate
    print(f"\n📊 Evaluating model...")
    test_metrics = model.evaluate(test_generator, verbose=0)
    
    print(f"✅ Test Results:")
    print(f"  Loss: {test_metrics[0]:.4f}")
    print(f"  Accuracy: {test_metrics[1]:.4f}")
    
    # Save model
    model_save_path = f'notebooks/saved_models/better_resnet50_model.h5'
    model.save(model_save_path)
    print(f"✅ Model saved to: {model_save_path}")
    
    # Plot training history
    print(f"\n📈 Plotting training history...")
    os.makedirs('results', exist_ok=True)
    plot_save_path = f'results/better_training_history.png'
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    results = {
        'model_type': 'ResNet50',
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'test_loss': test_metrics[0],
        'test_accuracy': test_metrics[1],
        'model_path': model_save_path,
        'training_history': history.history
    }
    
    results_save_path = f'results/better_training_results.json'
    import json
    with open(results_save_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n🎉 Training complete!")
    print(f"✅ Model saved to: {model_save_path}")
    print(f"✅ Results saved to: {results_save_path}")
    print(f"✅ Training plots saved to: {plot_save_path}")
    
    print(f"\n🚀 Next steps:")
    print(f"1. 🖥️  Launch GUI: python run_gui.py")
    print(f"2. 📊 Load your better model: {model_save_path}")
    print(f"3. 🖼️  Test with real X-ray images")
    
    print(f"\n💡 Key improvements implemented:")
    print(f"   • ResNet50 transfer learning (pre-trained on ImageNet)")
    print(f"   • Real RSNA pneumonia detection dataset with ImageDataGenerator")
    print(f"   • Proper train/validation/test splits with class subdirectories")
    print(f"   • Data augmentation and optimized hyperparameters")

if __name__ == "__main__":
    main()
