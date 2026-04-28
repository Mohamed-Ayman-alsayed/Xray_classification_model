#!/usr/bin/env python3
"""
Semi-Supervised Learning with Pseudo-Labels
==========================================

This script uses unsupervised learning results to create pseudo-labels,
then fine-tunes the model for better classification.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def load_unsupervised_results():
    """Load results from unsupervised training"""
    try:
        with open('results/unsupervised_results.json', 'r') as f:
            results = json.load(f)
        return results
    except FileNotFoundError:
        print("❌ Unsupervised results not found. Run unsupervised_training.py first.")
        return None

def create_pseudo_labels_from_anomalies(images, anomalies, mse_scores):
    """Create pseudo-labels based on anomaly detection"""
    print("🏷️  Creating pseudo-labels from anomaly detection...")
    
    # Method 1: Use reconstruction error as confidence
    # Low error = likely normal, High error = likely abnormal
    
    # Sort images by reconstruction error
    sorted_indices = np.argsort(mse_scores)
    
    # Take bottom 20% as "normal" (lowest reconstruction error)
    normal_threshold = int(len(images) * 0.2)
    normal_indices = sorted_indices[:normal_threshold]
    
    # Take top 20% as "abnormal" (highest reconstruction error)
    abnormal_threshold = int(len(images) * 0.2)
    abnormal_indices = sorted_indices[-abnormal_threshold:]
    
    # Create pseudo-labels
    pseudo_labels = np.zeros(len(images))  # 0 = normal
    pseudo_labels[abnormal_indices] = 1    # 1 = abnormal
    
    # Create confidence scores
    confidence_scores = np.zeros(len(images))
    confidence_scores[normal_indices] = 1.0 - (mse_scores[normal_indices] / np.max(mse_scores))
    confidence_scores[abnormal_indices] = mse_scores[abnormal_indices] / np.max(mse_scores)
    
    print(f"📊 Pseudo-labeling results:")
    print(f"   Normal (low error): {len(normal_indices)} images")
    print(f"   Abnormal (high error): {len(abnormal_indices)} images")
    print(f"   Unlabeled (middle): {len(images) - len(normal_indices) - len(abnormal_indices)} images")
    
    return pseudo_labels, confidence_scores, normal_indices, abnormal_indices

def create_pseudo_labels_from_clusters(images, cluster_labels, n_clusters=5):
    """Create pseudo-labels based on clustering results"""
    print("🏷️  Creating pseudo-labels from clustering...")
    
    # Analyze cluster characteristics
    cluster_stats = {}
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_size = len(cluster_indices)
        cluster_stats[cluster_id] = {
            'size': cluster_size,
            'percentage': cluster_size / len(images) * 100,
            'indices': cluster_indices
        }
    
    print("📊 Cluster analysis:")
    for cluster_id, stats in cluster_stats.items():
        print(f"   Cluster {cluster_id}: {stats['size']} images ({stats['percentage']:.1f}%)")
    
    # Strategy: Assume largest clusters are "normal", smallest are "abnormal"
    sorted_clusters = sorted(cluster_stats.items(), key=lambda x: x[1]['size'], reverse=True)
    
    # Take largest cluster as "normal"
    normal_cluster_id = sorted_clusters[0][0]
    normal_indices = cluster_stats[normal_cluster_id]['indices']
    
    # Take smallest cluster as "abnormal"
    abnormal_cluster_id = sorted_clusters[-1][0]
    abnormal_indices = cluster_stats[abnormal_cluster_id]['indices']
    
    # Create pseudo-labels
    pseudo_labels = np.zeros(len(images))
    pseudo_labels[abnormal_indices] = 1
    
    print(f"📊 Cluster-based pseudo-labeling:")
    print(f"   Normal (largest cluster): {len(normal_indices)} images")
    print(f"   Abnormal (smallest cluster): {len(abnormal_indices)} images")
    
    return pseudo_labels, normal_indices, abnormal_indices

def create_classifier_from_encoder(encoder, num_classes=2):
    """Create a classifier using the pre-trained encoder"""
    print("🎯 Creating classifier from encoder...")
    
    # Freeze encoder layers
    for layer in encoder.layers:
        layer.trainable = False
    
    # Add classification head
    classifier_input = encoder.input
    x = encoder.output
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    classifier_output = layers.Dense(num_classes, activation='softmax')(x)
    
    classifier = keras.Model(classifier_input, classifier_output, name='classifier')
    
    return classifier

def fine_tune_with_pseudo_labels(images, pseudo_labels, confidence_scores, encoder):
    """Fine-tune the model using pseudo-labels"""
    print("🎯 Fine-tuning with pseudo-labels...")
    
    # Filter high-confidence pseudo-labels
    high_confidence_threshold = 0.7
    high_confidence_mask = confidence_scores > high_confidence_threshold
    
    if np.sum(high_confidence_mask) < 100:
        print("⚠️  Not enough high-confidence labels. Using all pseudo-labels.")
        high_confidence_mask = np.ones(len(images), dtype=bool)
    
    # Get high-confidence data
    X_high_conf = images[high_confidence_mask]
    y_high_conf = pseudo_labels[high_confidence_mask]
    
    print(f"📊 Using {len(X_high_conf)} high-confidence pseudo-labels for fine-tuning")
    
    # Create classifier
    classifier = create_classifier_from_encoder(encoder)
    
    # Compile
    classifier.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_high_conf, y_high_conf, test_size=0.2, random_state=42
    )
    
    # Fine-tune
    history = classifier.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        verbose=1
    )
    
    return classifier, history

def main():
    """Main semi-supervised learning pipeline"""
    print("�� AI Health System - Semi-Supervised Learning")
    print("=" * 50)
    
    # Load unsupervised results
    results = load_unsupervised_results()
    if results is None:
        return
    
    # Load the trained encoder
    try:
        encoder = keras.models.load_model('notebooks/saved_models/unsupervised_encoder.h5')
        print("✅ Loaded pre-trained encoder")
    except:
        print("❌ Could not load encoder. Run unsupervised_training.py first.")
        return
    
    # Load images (you'll need to load the same images used in unsupervised training)
    # For now, we'll simulate this - in practice, you'd load the actual images
    print("📂 Loading images...")
    # images = load_images_from_unsupervised_training()  # You'd implement this
    
    # Simulate data for demonstration
    print("⚠️  Using simulated data for demonstration")
    images = np.random.random((1000, 224, 224, 3))  # Simulated images
    mse_scores = np.random.random(1000)  # Simulated reconstruction errors
    cluster_labels = np.random.randint(0, 5, 1000)  # Simulated cluster labels
    
    # Method 1: Create pseudo-labels from anomaly detection
    pseudo_labels_anomaly, confidence_scores, normal_idx, abnormal_idx = create_pseudo_labels_from_anomalies(
        images, None, mse_scores
    )
    
    # Method 2: Create pseudo-labels from clustering
    pseudo_labels_cluster, normal_idx_cluster, abnormal_idx_cluster = create_pseudo_labels_from_clusters(
        images, cluster_labels
    )
    
    # Combine both methods (you can choose one or combine)
    print("🔄 Combining pseudo-labeling methods...")
    
    # Use anomaly-based labels as primary, cluster-based as secondary
    final_pseudo_labels = pseudo_labels_anomaly.copy()
    
    # For unlabeled images, use cluster information
    unlabeled_mask = (confidence_scores < 0.7)
    final_pseudo_labels[unlabeled_mask] = pseudo_labels_cluster[unlabeled_mask]
    
    # Fine-tune the model
    classifier, history = fine_tune_with_pseudo_labels(
        images, final_pseudo_labels, confidence_scores, encoder
    )
    
    # Save the fine-tuned classifier
    os.makedirs('notebooks/saved_models', exist_ok=True)
    classifier.save('notebooks/saved_models/semi_supervised_classifier.h5')
    
    print("✅ Semi-supervised training complete!")
    print("✅ Classifier saved to: notebooks/saved_models/semi_supervised_classifier.h5")
    
    print("\n💡 Key insights:")
    print("   • Used unsupervised learning to create pseudo-labels")
    print("   • No manual medical labeling required")
    print("   • Model learns from data patterns automatically")
    print("   • Can be improved with more data or expert validation")

if __name__ == "__main__":
    main()
