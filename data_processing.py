"""
Data Processing Module for AI Health System
==========================================

Handles DICOM image loading, preprocessing, augmentation, and dataset management
for chest X-ray analysis.
"""

import os
import numpy as np
import cv2
import pydicom
from PIL import Image
import pandas as pd
from tqdm import tqdm
import random
from typing import Tuple, Dict, List, Optional, Union
import albumentations as A
from sklearn.model_selection import train_test_split

class ChestXRayProcessor:
    """Main processor for chest X-ray images"""
    
    def __init__(self, data_dir: str = '../notebooks/data', target_size: Tuple[int, int] = (224, 224)):
        self.data_dir = data_dir
        self.target_size = target_size
        self.raw_dir = os.path.join(data_dir, 'raw')
        self.processed_dir = os.path.join(data_dir, 'processed')
        
        # Create processed directories
        self.train_processed = os.path.join(self.processed_dir, 'train')
        self.test_processed = os.path.join(self.processed_dir, 'test')
        self.val_processed = os.path.join(self.processed_dir, 'validation')
        
        os.makedirs(self.train_processed, exist_ok=True)
        os.makedirs(self.test_processed, exist_ok=True)
        os.makedirs(self.val_processed, exist_ok=True)
        
        # Data augmentation pipeline
        self.augmentation_pipeline = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            A.OneOf([
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.3),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.1),
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.3),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2, p=0.3),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
                A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.3),
                A.RandomBrightnessContrast(p=0.3),
            ], p=0.3),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
        ])
    
    def scan_dataset(self) -> Dict[str, int]:
        """Scan dataset and return file counts by class"""
        print("🔍 Scanning dataset...")
        
        class_counts = {}
        train_raw = os.path.join(self.raw_dir, 'train')
        
        if os.path.exists(train_raw):
            for class_name in os.listdir(train_raw):
                class_path = os.path.join(train_raw, class_name)
                if os.path.isdir(class_path):
                    count = 0
                    for root, dirs, files in os.walk(class_path):
                        count += len([f for f in files if f.lower().endswith('.dcm')])
                    class_counts[class_name] = count
                    print(f"  📁 {class_name}: {count:,} files")
        
        return class_counts
    
    def load_dicom(self, file_path: str) -> Tuple[Optional[np.ndarray], Dict]:
        """Load DICOM file and return image array with metadata"""
        try:
            dicom_data = pydicom.dcmread(file_path)
            image_array = dicom_data.pixel_array
            
            # Get DICOM metadata
            metadata = {
                'patient_id': getattr(dicom_data, 'PatientID', 'Unknown'),
                'study_date': getattr(dicom_data, 'StudyDate', 'Unknown'),
                'modality': getattr(dicom_data, 'Modality', 'Unknown'),
                'body_part': getattr(dicom_data, 'BodyPartExamined', 'Unknown'),
                'shape': image_array.shape,
                'dtype': str(image_array.dtype),
                'min_val': float(image_array.min()),
                'max_val': float(image_array.max()),
                'file_size_mb': os.path.getsize(file_path) / (1024*1024)
            }
            
            return image_array, metadata
            
        except Exception as e:
            return None, {'error': str(e)}
    
    def preprocess_image(self, image: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Preprocess image: resize, normalize, convert to RGB if needed"""
        # Resize to target size
        image_resized = cv2.resize(image, self.target_size)
        
        # Normalize to [0, 1] range
        if normalize:
            if image_resized.max() > 1:
                image_resized = image_resized.astype(np.float32) / 255.0
        
        # Ensure 3 channels for RGB models
        if len(image_resized.shape) == 2:
            image_resized = np.stack([image_resized] * 3, axis=-1)
        
        return image_resized
    
    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """Apply data augmentation to image"""
        augmented = self.augmentation_pipeline(image=image)
        return augmented['image']
    
    def process_dataset(self, source_dir: str, output_dir: str, 
                       max_samples: Optional[int] = None, 
                       augment: bool = True) -> pd.DataFrame:
        """Process entire dataset with preprocessing and optional augmentation"""
        print(f"🔄 Processing dataset from {source_dir} to {output_dir}")
        
        processed_files = []
        all_files = []
        
        # Collect all DICOM files
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith('.dcm'):
                    all_files.append(os.path.join(root, file))
        
        if max_samples:
            all_files = all_files[:max_samples]
        
        print(f"📊 Found {len(all_files):,} files to process")
        
        # Process each file
        for file_path in tqdm(all_files, desc="Processing images"):
            try:
                # Load DICOM
                image_array, metadata = self.load_dicom(file_path)
                if image_array is None:
                    continue
                
                # Preprocess
                processed_image = self.preprocess_image(image_array)
                
                # Save processed image
                filename = os.path.splitext(os.path.basename(file_path))[0]
                output_path = os.path.join(output_dir, f"{filename}.png")
                
                # Convert to PIL and save
                if processed_image.max() <= 1:
                    processed_image = (processed_image * 255).astype(np.uint8)
                
                Image.fromarray(processed_image).save(output_path)
                
                # Record metadata
                processed_files.append({
                    'original_path': file_path,
                    'processed_path': output_path,
                    'patient_id': metadata.get('patient_id', 'Unknown'),
                    'study_date': metadata.get('study_date', 'Unknown'),
                    'original_shape': metadata.get('shape', (0, 0)),
                    'processed_shape': self.target_size,
                    'file_size_mb': metadata.get('file_size_mb', 0)
                })
                
                # Apply augmentation if requested
                if augment:
                    for i in range(2):  # Create 2 augmented versions
                        augmented_image = self.augment_image(processed_image)
                        aug_filename = f"{filename}_aug_{i}.png"
                        aug_output_path = os.path.join(output_dir, aug_filename)
                        
                        if augmented_image.max() <= 1:
                            augmented_image = (augmented_image * 255).astype(np.uint8)
                        
                        Image.fromarray(augmented_image).save(aug_output_path)
                        
                        processed_files.append({
                            'original_path': file_path,
                            'processed_path': aug_output_path,
                            'patient_id': metadata.get('patient_id', 'Unknown'),
                            'study_date': metadata.get('study_date', 'Unknown'),
                            'original_shape': metadata.get('shape', (0, 0)),
                            'processed_shape': self.target_size,
                            'file_size_mb': metadata.get('file_size_mb', 0),
                            'augmented': True
                        })
                
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
                continue
        
        # Save processing log
        df = pd.DataFrame(processed_files)
        log_path = os.path.join(output_dir, 'processing_log.csv')
        df.to_csv(log_path, index=False)
        
        print(f"✅ Processing complete! {len(processed_files):,} files processed")
        print(f"📝 Log saved to: {log_path}")
        
        return df
    
    def create_train_val_split(self, train_dir: str, val_ratio: float = 0.2) -> None:
        """Create train/validation split from processed training data"""
        print(f"✂️  Creating train/validation split (val_ratio={val_ratio})")
        
        # Get all processed images
        train_images = []
        for file in os.listdir(train_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                train_images.append(file)
        
        # Split
        train_files, val_files = train_test_split(
            train_images, test_size=val_ratio, random_state=42
        )
        
        # Move validation files
        for file in val_files:
            src_path = os.path.join(train_dir, file)
            dst_path = os.path.join(self.val_processed, file)
            os.rename(src_path, dst_path)
        
        print(f"✅ Split complete: {len(train_files)} train, {len(val_files)} validation")
    
    def get_class_distribution(self, directory: str) -> Dict[str, int]:
        """Get class distribution from directory structure"""
        class_counts = {}
        
        if os.path.exists(directory):
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isdir(item_path):
                    # Count files in subdirectory
                    count = len([f for f in os.listdir(item_path) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    class_counts[item] = count
        
        return class_counts

def load_real_dataset(data_dir: str = '../notebooks/data', img_size: Tuple[int, int] = (224, 224), max_samples: Optional[int] = None):
    """Load real RSNA pneumonia detection dataset"""
    print("🏥 Loading real RSNA pneumonia detection dataset...")
    
    import pandas as pd
    from PIL import Image
    from tensorflow.keras.utils import to_categorical
    
    # Load labels
    labels_path = os.path.join(data_dir, 'reports', 'stage_2_detailed_class_info.csv')
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    labels_df = pd.read_csv(labels_path)
    print(f"📋 Loaded {len(labels_df)} label records")
    
    # Map classes to binary labels
    # Normal = 0, Lung Opacity = 1, No Lung Opacity / Not Normal = 0 (treating as normal)
    def map_class_to_binary(class_name):
        if class_name == 'Lung Opacity':
            return 1  # Abnormal (pneumonia)
        else:
            return 0  # Normal (no pneumonia)
    
    labels_df['binary_label'] = labels_df['class'].apply(map_class_to_binary)
    
    # Get class distribution
    class_counts = labels_df['binary_label'].value_counts()
    print(f"📊 Class distribution:")
    print(f"  Normal (0): {class_counts.get(0, 0)} samples")
    print(f"  Abnormal (1): {class_counts.get(1, 0)} samples")
    
    # Load images from processed directory
    processed_dir = os.path.join(data_dir, 'processed')
    train_dir = os.path.join(processed_dir, 'train')
    val_dir = os.path.join(processed_dir, 'validation')
    test_dir = os.path.join(processed_dir, 'test')
    
    def load_images_from_directory(directory, labels_df, max_samples=None):
        """Load images from a directory with their corresponding labels"""
        images = []
        labels = []
        patient_ids = []
        
        if not os.path.exists(directory):
            print(f"⚠️  Directory not found: {directory}")
            return np.array([]), np.array([]), []
        
        # Get all image files
        image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if max_samples:
            image_files = image_files[:max_samples]
        
        print(f"🖼️  Loading {len(image_files)} images from {directory}")
        
        for img_file in tqdm(image_files, desc=f"Loading from {os.path.basename(directory)}"):
            # Extract patient ID from filename (remove extension)
            patient_id = os.path.splitext(img_file)[0]
            
            # Find corresponding label
            label_row = labels_df[labels_df['patientId'] == patient_id]
            
            if not label_row.empty:
                # Load and preprocess image
                img_path = os.path.join(directory, img_file)
                try:
                    img = Image.open(img_path)
                    img = img.convert('RGB')
                    img = img.resize(img_size)
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    
                    images.append(img_array)
                    labels.append(label_row['binary_label'].iloc[0])
                    patient_ids.append(patient_id)
                    
                except Exception as e:
                    print(f"⚠️  Error loading {img_path}: {e}")
                    continue
        
        return np.array(images), np.array(labels), patient_ids
    
    # Load training data
    X_train, y_train, train_ids = load_images_from_directory(train_dir, labels_df, max_samples)
    
    # Load validation data
    X_val, y_val, val_ids = load_images_from_directory(val_dir, labels_df, max_samples)
    
    # Load test data
    X_test, y_test, test_ids = load_images_from_directory(test_dir, labels_df, max_samples)
    
    # Convert labels to categorical
    if len(y_train) > 0:
        y_train = to_categorical(y_train, 2)
    if len(y_val) > 0:
        y_val = to_categorical(y_val, 2)
    if len(y_test) > 0:
        y_test = to_categorical(y_test, 2)
    
    print(f"✅ Real dataset loaded:")
    print(f"  📊 Training: {len(X_train)} images")
    print(f"  📊 Validation: {len(X_val)} images")
    print(f"  📊 Test: {len(X_test)} images")
    
    if len(y_train) > 0:
        print(f"  📊 Training class distribution: {np.sum(y_train, axis=0)}")
    if len(y_val) > 0:
        print(f"  📊 Validation class distribution: {np.sum(y_val, axis=0)}")
    if len(y_test) > 0:
        print(f"  📊 Test class distribution: {np.sum(y_test, axis=0)}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def create_processing_pipeline(data_dir: str = '../notebooks/data', target_size: Tuple[int, int] = (224, 224)) -> ChestXRayProcessor:
    """Factory function to create processor with standard settings"""
    return ChestXRayProcessor(data_dir=data_dir, target_size=target_size)
