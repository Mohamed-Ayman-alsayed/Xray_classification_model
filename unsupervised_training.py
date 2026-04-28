#!/usr/bin/env python3
"""
Unsupervised Learning for Unlabeled Medical Images
================================================

This script implements unsupervised learning techniques for unlabeled chest X-ray data:
- Autoencoders for anomaly detection
- Clustering for image grouping
- Self-supervised learning for representation learning
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import cv2
from PIL import Image
import glob

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_autoencoder(input_shape=(224, 224, 3), encoding_dim=128):
    """Create an autoencoder for anomaly detection"""
    
    # Encoder
    encoder_input = keras.Input(shape=input_shape, name='encoder_input')
    
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(encoder_input)
    x = layers.MaxPooling2D(2, padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2, padding="same")(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2, padding="same")(x)
    x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2, padding="same")(x)
    
    # Flatten and encode
    x = layers.Flatten()(x)
    encoded = layers.Dense(encoding_dim, activation="relu", name='encoded')(x)
    
    # # Decoder
    # x = layers.Dense(7 * 7 * 256, activation="relu")(encoded)
    # x = layers.Reshape((7, 7, 256))(x)
    
    # x = layers.Conv2DTranspose(128, 3, activation="relu", padding="same")(x)
    # x = layers.UpSampling2D(2)(x)
    # x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
    # x = layers.UpSampling2D(2)(x)
    # x = layers.Conv2DTranspose(32, 3, activation="relu", padding="same")(x)
    # x = layers.UpSampling2D(2)(x)
    # x = layers.Conv2DTranspose(16, 3, activation="relu", padding="same")(x)
    # x = layers.UpSampling2D(2)(x)
    # decoded = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
        # Decoder (fixed to return 224x224x3)
    x = layers.Dense(14 * 14 * 256, activation="relu")(encoded)
    x = layers.Reshape((14, 14, 256))(x)

    x = layers.Conv2DTranspose(128, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling2D(2)(x)   # 14 -> 28
    x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling2D(2)(x)   # 28 -> 56
    x = layers.Conv2DTranspose(32, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling2D(2)(x)   # 56 -> 112
    x = layers.Conv2DTranspose(16, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling2D(2)(x)   # 112 -> 224

    decoded = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
    
    autoencoder = keras.Model(encoder_input, decoded, name='autoencoder')
    encoder = keras.Model(encoder_input, encoded, name='encoder')
    
    return autoencoder, encoder

def load_unlabeled_data(data_dir, max_samples=None, target_size=(224, 224)):
    """Load unlabeled images from directory"""
    print(f"📂 Loading unlabeled data from: {data_dir}")
    
    # Find all image files
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.dcm']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(data_dir, '**', ext), recursive=True))
    
    if max_samples:
        image_paths = image_paths[:max_samples]
    
    print(f"📊 Found {len(image_paths)} images")
    
    images = []
    valid_paths = []
    
    for i, path in enumerate(image_paths):
        try:
            if path.endswith('.dcm'):
                # Handle DICOM files
                import pydicom
                ds = pydicom.dcmread(path)
                img_array = ds.pixel_array
                img_array = cv2.resize(img_array, target_size)
                if len(img_array.shape) == 2:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            else:
                # Handle regular image files
                img = Image.open(path).convert('RGB')
                img = img.resize(target_size)
                img_array = np.array(img)
            
            # Normalize to [0, 1]
            img_array = img_array.astype(np.float32) / 255.0
            images.append(img_array)
            valid_paths.append(path)
            
            if (i + 1) % 1000 == 0:
                print(f"   Processed {i + 1}/{len(image_paths)} images")
                
        except Exception as e:
            print(f"   ⚠️  Skipping {path}: {e}")
            continue
    
    print(f"✅ Successfully loaded {len(images)} images")
    return np.array(images), valid_paths

def train_autoencoder(images, epochs=50, batch_size=32):
    """Train autoencoder on unlabeled data"""
    print(f"\n🎯 Training Autoencoder...")
    print(f"   Images: {images.shape}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    
    # Create autoencoder
    autoencoder, encoder = create_autoencoder()
    
    # Compile
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # Train
    history = autoencoder.fit(
        images, images,  # Input and target are the same (reconstruction)
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )
    
    return autoencoder, encoder, history

def detect_anomalies(autoencoder, images, threshold_percentile=95):
    """Detect anomalies using reconstruction error"""
    print(f"\n🔍 Detecting anomalies...")
    
    # Get reconstructions
    reconstructions = autoencoder.predict(images, verbose=0)
    
    # Calculate reconstruction errors
    mse = np.mean(np.square(images - reconstructions), axis=(1, 2, 3))
    
    # Set threshold
    threshold = np.percentile(mse, threshold_percentile)
    
    # Find anomalies
    anomalies = mse > threshold
    anomaly_indices = np.where(anomalies)[0]
    
    print(f"📊 Results:")
    print(f"   Total images: {len(images)}")
    print(f"   Anomalies detected: {len(anomaly_indices)} ({len(anomaly_indices)/len(images)*100:.1f}%)")
    print(f"   Threshold (MSE): {threshold:.6f}")
    
    return anomalies, mse, threshold

def cluster_images(encoder, images, n_clusters=5):
    """Cluster images using encoded features"""
    print(f"\n🎯 Clustering images into {n_clusters} groups...")
    
    # Extract features
    features = encoder.predict(images, verbose=0)
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(features)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features_pca)
    
    print(f"�� Clustering results:")
    for i in range(n_clusters):
        count = np.sum(cluster_labels == i)
        print(f"   Cluster {i}: {count} images ({count/len(images)*100:.1f}%)")
    
    return cluster_labels, features_pca

def visualize_results(images, anomalies, cluster_labels, mse, save_dir='results'):
    """Create visualizations of results"""
    print(f"\n📈 Creating visualizations...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot reconstruction errors
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(mse, bins=50, alpha=0.7, color='blue')
    plt.axvline(np.percentile(mse, 95), color='red', linestyle='--', label='95th percentile')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Reconstruction Errors')
    plt.legend()
    
    # Show some anomalies
    plt.subplot(2, 2, 2)
    anomaly_indices = np.where(anomalies)[0][:6]  # Show first 6 anomalies
    for i, idx in enumerate(anomaly_indices):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[idx])
        plt.title(f'Anomaly {i+1}\nMSE: {mse[idx]:.4f}')
        plt.axis('off')
    
    # Show cluster examples
    plt.subplot(2, 2, 3)
    for cluster_id in range(min(5, len(np.unique(cluster_labels)))):
        cluster_indices = np.where(cluster_labels == cluster_id)[0][:3]
        for i, idx in enumerate(cluster_indices):
            plt.subplot(3, 5, cluster_id*3 + i + 1)
            plt.imshow(images[idx])
            plt.title(f'C{cluster_id}')
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'unsupervised_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualizations saved to: {save_dir}/unsupervised_results.png")

def main():
    """Main unsupervised learning pipeline"""
    print("�� AI Health System - Unsupervised Learning")
    print("=" * 50)
    
    # Configuration
    # Build a cross-platform path to the training data directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "notebooks", "data", "processed", "train")  # Unlabeled processed images
    max_samples = 2500  # Limit for faster processing
    epochs = 30
    batch_size = 32
    n_clusters = 5
    
    print(f"⚙️  Configuration:")
    print(f"   Data directory: {data_dir}")
    print(f"   Max samples: {max_samples}")
    print(f"   Epochs: {epochs}")
    print(f"   Clusters: {n_clusters}")
    
    # Load unlabeled data
    images, image_paths = load_unlabeled_data(data_dir, max_samples)
    
    if len(images) == 0:
        print("❌ No images found! Check your data directory.")
        return
    
    # Train autoencoder
    autoencoder, encoder, history = train_autoencoder(images, epochs, batch_size)
    
    # Save models
    os.makedirs('notebooks/saved_models', exist_ok=True)
    autoencoder.save('notebooks/saved_models/unsupervised_autoencoder.h5')
    encoder.save('notebooks/saved_models/unsupervised_encoder.h5')
    print(f"✅ Models saved to: notebooks/saved_models/")
    
    # Detect anomalies
    anomalies, mse, threshold = detect_anomalies(autoencoder, images)
    
    # Cluster images
    cluster_labels, features_pca = cluster_images(encoder, images, n_clusters)
    
    # Create visualizations
    visualize_results(images, anomalies, cluster_labels, mse)
    
    # Save results
    results = {
        'total_images': len(images),
        'anomalies_detected': int(np.sum(anomalies)),
        'anomaly_percentage': float(np.sum(anomalies) / len(images) * 100),
        'threshold_mse': float(threshold),
        'cluster_distribution': {f'cluster_{i}': int(np.sum(cluster_labels == i)) 
                               for i in range(n_clusters)},
        'model_paths': {
            'autoencoder': 'notebooks/saved_models/unsupervised_autoencoder.h5',
            'encoder': 'notebooks/saved_models/unsupervised_encoder.h5'
        }
    }
    
    import json
    with open('results/unsupervised_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎉 Unsupervised learning complete!")
    print(f"✅ Results saved to: results/unsupervised_results.json")
    print(f"✅ Models saved to: notebooks/saved_models/")
    
    print(f"\n💡 Key findings:")
    print(f"   • {results['anomalies_detected']} anomalous images detected ({results['anomaly_percentage']:.1f}%)")
    print(f"   • Images clustered into {n_clusters} groups")
    print(f"   • Autoencoder trained for anomaly detection")
    
    print(f"\n🚀 Next steps:")
    print(f"1. 🖥️  Launch GUI: python run_gui.py")
    print(f"2. 🔍 Load autoencoder model for anomaly detection")
    print(f"3. 📊 Review clustering results and visualizations")

if __name__ == "__main__":
    main()
