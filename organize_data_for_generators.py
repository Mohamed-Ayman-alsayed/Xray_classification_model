#!/usr/bin/env python3
"""
Organize Data for ImageDataGenerator
===================================

This script organizes the RSNA dataset into class subdirectories
so that ImageDataGenerator can work properly.
"""

import os
import pandas as pd
import shutil
from pathlib import Path

def organize_data_for_generators(data_dir='../notebooks/data'):
    """Organize data into class subdirectories for ImageDataGenerator"""
    print("🗂️  Organizing data for ImageDataGenerator...")
    
    # Load labels
    labels_path = os.path.join(data_dir, 'reports', 'stage_2_detailed_class_info.csv')
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    labels_df = pd.read_csv(labels_path)
    print(f"📋 Loaded {len(labels_df)} label records")
    
    # Map classes to binary labels
    def map_class_to_binary(class_name):
        if class_name == 'Lung Opacity':
            return 'abnormal'  # Pneumonia
        else:
            return 'normal'    # No pneumonia
    
    labels_df['class_folder'] = labels_df['class'].apply(map_class_to_binary)
    
    # Get class distribution
    class_counts = labels_df['class_folder'].value_counts()
    print(f"📊 Class distribution:")
    print(f"  Normal: {class_counts.get('normal', 0)} samples")
    print(f"  Abnormal: {class_counts.get('abnormal', 0)} samples")
    
    # Process each split
    splits = ['train', 'validation', 'test']
    
    for split in splits:
        print(f"\n🔄 Processing {split} split...")
        
        # Source directory
        source_dir = os.path.join(data_dir, 'processed', split)
        if not os.path.exists(source_dir):
            print(f"⚠️  Source directory not found: {source_dir}")
            continue
        
        # Create organized directory structure
        organized_dir = os.path.join(data_dir, 'organized', split)
        normal_dir = os.path.join(organized_dir, 'normal')
        abnormal_dir = os.path.join(organized_dir, 'abnormal')
        
        os.makedirs(normal_dir, exist_ok=True)
        os.makedirs(abnormal_dir, exist_ok=True)
        
        # Get all image files
        image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"📁 Found {len(image_files)} images in {split}")
        
        # Copy files to appropriate class directories
        copied_count = 0
        for img_file in image_files:
            # Extract patient ID from filename (remove extension)
            patient_id = os.path.splitext(img_file)[0]
            
            # Find corresponding label
            label_row = labels_df[labels_df['patientId'] == patient_id]
            
            if not label_row.empty:
                class_folder = label_row['class_folder'].iloc[0]
                
                # Source and destination paths
                src_path = os.path.join(source_dir, img_file)
                dst_dir = os.path.join(organized_dir, class_folder)
                dst_path = os.path.join(dst_dir, img_file)
                
                try:
                    shutil.copy2(src_path, dst_path)
                    copied_count += 1
                except Exception as e:
                    print(f"⚠️  Error copying {img_file}: {e}")
                    continue
        
        print(f"✅ Copied {copied_count} images to organized {split} directory")
        
        # Count files in each class
        normal_count = len([f for f in os.listdir(normal_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        abnormal_count = len([f for f in os.listdir(abnormal_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        print(f"  📊 Normal: {normal_count} images")
        print(f"  📊 Abnormal: {abnormal_count} images")
    
    print(f"\n✅ Data organization complete!")
    print(f"📁 Organized data saved to: {os.path.join(data_dir, 'organized')}")
    print(f"📁 Structure:")
    print(f"  organized/")
    print(f"  ├── train/")
    print(f"  │   ├── normal/")
    print(f"  │   └── abnormal/")
    print(f"  ├── validation/")
    print(f"  │   ├── normal/")
    print(f"  │   └── abnormal/")
    print(f"  └── test/")
    print(f"      ├── normal/")
    print(f"      └── abnormal/")

if __name__ == "__main__":
    organize_data_for_generators()
