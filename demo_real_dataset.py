#!/usr/bin/env python3
"""
Demo: Real Dataset vs Synthetic Data
===================================

This script demonstrates the difference between synthetic data generation
and real dataset loading for the AI Health System.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

from data_processing import load_real_dataset

def create_synthetic_data_demo(num_samples=10, img_size=(224, 224)):
    """Create a small sample of synthetic data for comparison"""
    print("🎲 Creating synthetic data sample...")
    
    X = np.zeros((num_samples, *img_size, 3), dtype=np.float32)
    
    for i in range(num_samples):
        # Create base with chest X-ray characteristics
        base = np.random.normal(0.4, 0.15, img_size)
        
        # Add structural elements
        # Horizontal lines (ribs)
        for rib in range(2, 9):
            y_pos = int(img_size[1] * rib / 10)
            base[y_pos-1:y_pos+1, :] += np.random.normal(0.1, 0.03)
        
        # Vertical structure (spine)
        spine_x = img_size[0] // 2
        base[:, spine_x-2:spine_x+2] += np.random.normal(0.12, 0.04)
        
        # Add some circular areas (lungs)
        center_y, center_x = img_size[1] // 2, img_size[0] // 2
        y_coords, x_coords = np.ogrid[:img_size[1], :img_size[0]]
        
        # Left lung
        left_lung = ((x_coords - center_x + 30)**2 + (y_coords - center_y)**2) < 60**2
        base[left_lung] += np.random.normal(0.05, 0.02)
        
        # Right lung
        right_lung = ((x_coords - center_x - 30)**2 + (y_coords - center_y)**2) < 60**2
        base[right_lung] += np.random.normal(0.05, 0.02)
        
        # Add noise
        noise = np.random.normal(0, 0.03, img_size)
        base += noise
        
        # Normalize
        base = np.clip(base, 0, 1)
        
        # Stack to RGB
        X[i] = np.stack([base] * 3, axis=-1)
    
    # Create random labels
    y = np.random.randint(0, 2, num_samples)
    
    return X, y

def main():
    """Main demonstration function"""
    print("🚀 AI Health System - Real Dataset vs Synthetic Data Demo")
    print("=" * 60)
    
    # Load real dataset (small sample)
    print("\n🏥 Loading real RSNA dataset...")
    try:
        (X_train_real, y_train_real), (X_val_real, y_val_real), (X_test_real, y_test_real) = load_real_dataset(
            data_dir='../notebooks/data', 
            img_size=(224, 224), 
            max_samples=5  # Small sample for demo
        )
        
        print(f"✅ Real dataset loaded successfully!")
        print(f"  Training samples: {len(X_train_real)}")
        print(f"  Validation samples: {len(X_val_real)}")
        print(f"  Test samples: {len(X_test_real)}")
        
        if len(y_train_real) > 0:
            print(f"  Training class distribution: {np.sum(y_train_real, axis=0)}")
        
    except Exception as e:
        print(f"❌ Error loading real dataset: {e}")
        return
    
    # Create synthetic data for comparison
    print("\n🎲 Creating synthetic data for comparison...")
    X_synthetic, y_synthetic = create_synthetic_data_demo(num_samples=5)
    print(f"✅ Synthetic data created!")
    print(f"  Samples: {len(X_synthetic)}")
    print(f"  Class distribution: {np.bincount(y_synthetic)}")
    
    # Create comparison visualization
    print("\n📊 Creating comparison visualization...")
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Real Dataset vs Synthetic Data Comparison', fontsize=16)
    
    # Plot real data
    if len(X_train_real) > 0:
        for i in range(min(5, len(X_train_real))):
            axes[0, i].imshow(X_train_real[i])
            axes[0, i].set_title(f'Real - Class {np.argmax(y_train_real[i])}')
            axes[0, i].axis('off')
    else:
        for i in range(5):
            axes[0, i].text(0.5, 0.5, 'No Real Data', ha='center', va='center')
            axes[0, i].axis('off')
    
    # Plot synthetic data
    for i in range(5):
        axes[1, i].imshow(X_synthetic[i])
        axes[1, i].set_title(f'Synthetic - Class {y_synthetic[i]}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    # Save the comparison
    os.makedirs('../results', exist_ok=True)
    plt.savefig('../results/real_vs_synthetic_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Comparison saved to: ../results/real_vs_synthetic_comparison.png")
    
    # Print summary
    print("\n📋 Summary:")
    print("=" * 40)
    print("🏥 Real Dataset:")
    print("  • Actual chest X-ray images from RSNA dataset")
    print("  • Real pneumonia detection labels")
    print("  • Proper train/validation/test splits")
    print("  • Realistic class distribution")
    print("  • High-quality medical imaging data")
    
    print("\n🎲 Synthetic Data:")
    print("  • Artificially generated images")
    print("  • Simulated chest X-ray features")
    print("  • Random labels")
    print("  • Good for testing but not for real training")
    
    print("\n💡 Recommendation:")
    print("  Use the real dataset for actual model training!")
    print("  The synthetic data was just for initial testing.")

if __name__ == "__main__":
    main()
