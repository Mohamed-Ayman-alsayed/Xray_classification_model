"""
Utility Functions for AI Health System
====================================

Helper functions for data visualization, metrics calculation, and file operations.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from typing import List, Tuple, Dict, Any, Optional
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def plot_training_history(history: Any, save_path: str = None, 
                         figsize: Tuple[int, int] = (15, 10)) -> None:
    """Plot comprehensive training history from Keras model"""
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    if 'val_accuracy' in history.history:
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
    if 'val_loss' in history.history:
        axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Precision
    if 'precision' in history.history:
        axes[2].plot(history.history['precision'], label='Training Precision', linewidth=2)
        if 'val_precision' in history.history:
            axes[2].plot(history.history['val_precision'], label='Validation Precision', linewidth=2)
        axes[2].set_title('Model Precision', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Precision')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    # Recall
    if 'recall' in history.history:
        axes[3].plot(history.history['recall'], label='Training Recall', linewidth=2)
        if 'val_recall' in history.history:
            axes[3].plot(history.history['val_recall'], label='Validation Recall', linewidth=2)
        axes[3].set_title('Model Recall', fontsize=14, fontweight='bold')
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('Recall')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
    
    # AUC
    if 'auc' in history.history:
        axes[4].plot(history.history['auc'], label='Training AUC', linewidth=2)
        if 'val_auc' in history.history:
            axes[4].plot(history.history['val_auc'], label='Validation AUC', linewidth=2)
        axes[4].set_title('Model AUC', fontsize=14, fontweight='bold')
        axes[4].set_xlabel('Epoch')
        axes[4].set_ylabel('AUC')
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)
    
    # Learning Rate (if available)
    if 'lr' in history.history:
        axes[5].plot(history.history['lr'], label='Learning Rate', linewidth=2)
        axes[5].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[5].set_xlabel('Epoch')
        axes[5].set_ylabel('Learning Rate')
        axes[5].set_yscale('log')
        axes[5].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training history plot saved to: {save_path}")
    
    plt.show()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: List[str] = None, save_path: str = None,
                         normalize: bool = True) -> None:
    """Plot confusion matrix with enhanced visualization"""
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized Confusion Matrix'
        fmt = '.2f'
    else:
        title = 'Confusion Matrix'
        fmt = 'd'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names or ['Normal', 'Pneumonia'],
                yticklabels=class_names or ['Normal', 'Pneumonia'])
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to: {save_path}")
    
    plt.show()

def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                   class_names: List[str] = None, save_path: str = None) -> None:
    """Plot ROC curve for binary classification"""
    if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
        # Multi-class: use one-vs-rest
        n_classes = y_pred_proba.shape[1]
        plt.figure(figsize=(10, 8))
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            class_name = class_names[i] if class_names else f'Class {i}'
            plt.plot(fpr, tpr, linewidth=2, 
                    label=f'{class_name} (AUC = {roc_auc:.3f})')
    else:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, 
                label=f'ROC Curve (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ ROC curve saved to: {save_path}")
    
    plt.show()

def plot_class_distribution(class_counts: Dict[str, int], save_path: str = None) -> None:
    """Plot class distribution in dataset"""
    plt.figure(figsize=(10, 6))
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    bars = plt.bar(classes, counts, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
    plt.title('Dataset Class Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Class distribution plot saved to: {save_path}")
    
    plt.show()

def plot_sample_images(images: List[np.ndarray], labels: List[str], 
                      class_names: List[str] = None, save_path: str = None,
                      figsize: Tuple[int, int] = (15, 10)) -> None:
    """Plot sample images with labels"""
    n_images = len(images)
    n_cols = min(5, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (image, label) in enumerate(zip(images, labels)):
        row = i // n_cols
        col = i % n_cols
        
        if n_rows == 1:
            ax = axes[col] if n_cols > 1 else axes
        elif n_cols == 1:
            ax = axes[row] if n_rows > 1 else axes
        else:
            ax = axes[row, col]
        
        ax.imshow(image, cmap='gray')
        ax.set_title(f'{label}', fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(n_images, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if n_rows == 1:
            ax = axes[col] if n_cols > 1 else axes
        elif n_cols == 1:
            ax = axes[row] if n_rows > 1 else axes
        else:
            ax = axes[row, col]
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Sample images plot saved to: {save_path}")
    
    plt.show()

def create_directory_structure(base_path: str) -> None:
    """Create the standard directory structure for the project"""
    directories = [
        'notebooks/data/raw/train',
        'notebooks/data/raw/test',
        'notebooks/data/processed/train',
        'notebooks/data/processed/test',
        'notebooks/data/processed/validation',
        'notebooks/data/reports',
        'notebooks/saved_models',
        'logs',
        'results',
        'plots'
    ]
    
    for directory in directories:
        full_path = os.path.join(base_path, directory)
        os.makedirs(full_path, exist_ok=True)
        print(f"✅ Created directory: {full_path}")

def save_results(results: Dict[str, Any], filepath: str) -> None:
    """Save results to a JSON file"""
    import json
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✅ Results saved to: {filepath}")

def load_results(filepath: str) -> Dict[str, Any]:
    """Load results from a JSON file"""
    import json
    
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    print(f"✅ Results loaded from: {filepath}")
    return results

def print_dataset_info(data_dir: str) -> None:
    """Print comprehensive information about the dataset structure"""
    print("📊 DATASET INFORMATION")
    print("=" * 60)
    
    if os.path.exists(data_dir):
        total_files = 0
        class_info = {}
        
        for root, dirs, files in os.walk(data_dir):
            level = root.replace(data_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            
            if level == 0:
                print(f"{indent}📁 {os.path.basename(root)}/")
            else:
                file_count = len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm'))])
                if file_count > 0:
                    print(f"{indent}📁 {os.path.basename(root)}/ ({file_count} files)")
                    total_files += file_count
                    
                    # Track class information
                    if level == 1:  # Class level
                        class_name = os.path.basename(root)
                        class_info[class_name] = file_count
        
        print(f"\n📈 SUMMARY:")
        print(f"  Total files: {total_files:,}")
        if class_info:
            print(f"  Classes: {len(class_info)}")
            for class_name, count in class_info.items():
                print(f"    {class_name}: {count:,} files")
    else:
        print("❌ Data directory not found!")

def calculate_model_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                           y_pred_proba: np.ndarray = None) -> Dict[str, float]:
    """Calculate comprehensive model metrics"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }
    
    if y_pred_proba is not None:
        try:
            from sklearn.metrics import roc_auc_score
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, average='weighted', multi_class='ovr')
        except:
            pass
    
    return metrics

def print_model_summary(metrics: Dict[str, float]) -> None:
    """Print formatted model performance summary"""
    print("\n🎯 MODEL PERFORMANCE SUMMARY")
    print("=" * 40)
    
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
        else:
            print(f"  {metric.replace('_', ' ').title()}: {value}")
    
    print("=" * 40)
