#!/usr/bin/env python3
"""
Check Data Structure Script
===========================

This script checks the current data structure and shows what's available.
"""

import os
from pathlib import Path

def check_data_structure():
    """Check the current data structure"""
    print("🔍 Checking AI Health System Data Structure")
    print("=" * 60)
    
    # Check base directories
    base_dirs = [
        'notebooks/data',
        'notebooks/saved_models',
        'src',
        'logs',
        'results'
    ]
    
    for base_dir in base_dirs:
        if os.path.exists(base_dir):
            print(f"✅ {base_dir}/")
        else:
            print(f"❌ {base_dir}/ (missing)")
    
    print("\n📁 Detailed Data Structure:")
    print("-" * 40)
    
    # Check notebooks/data structure
    data_dir = Path('notebooks/data')
    if data_dir.exists():
        print(f"📂 {data_dir}/")
        
        # Check subdirectories
        for item in data_dir.iterdir():
            if item.is_dir():
                print(f"  📁 {item.name}/")
                
                # Check contents of each subdirectory
                if item.name == 'raw':
                    raw_contents = list(item.iterdir())
                    for raw_item in raw_contents:
                        if raw_item.is_dir():
                            file_count = len(list(raw_item.glob('*.dcm')))
                            print(f"    📂 {raw_item.name}/ ({file_count} DICOM files)")
                        else:
                            print(f"    📄 {raw_item.name}")
                
                elif item.name == 'processed':
                    proc_contents = list(item.iterdir())
                    for proc_item in proc_contents:
                        if proc_item.is_dir():
                            file_count = len(list(proc_item.glob('*.png')))
                            print(f"    📂 {proc_item.name}/ ({file_count} PNG files)")
                        else:
                            print(f"    📄 {proc_item.name}")
                
                elif item.name == 'reports':
                    report_files = list(item.glob('*'))
                    if report_files:
                        for report_file in report_files:
                            print(f"    📄 {report_file.name}")
                    else:
                        print("    📄 (empty)")
                
                else:
                    # Count files in other directories
                    file_count = len(list(item.iterdir()))
                    print(f"    📄 ({file_count} items)")
    else:
        print("❌ notebooks/data/ directory not found!")
    
    print("\n🔧 Configuration:")
    print("-" * 40)
    
    # Check if config.py exists and show data directory
    if os.path.exists('config.py'):
        try:
            import config
            print(f"✅ Data directory configured: {config.DATA_DIR}")
            print(f"✅ Models directory configured: {config.MODELS_DIR}")
        except Exception as e:
            print(f"⚠️  Config file exists but has errors: {e}")
    else:
        print("❌ config.py not found")
    
    print("\n📊 Summary:")
    print("-" * 40)
    
    # Count total files
    total_dicom = 0
    total_processed = 0
    
    if data_dir.exists():
        # Count DICOM files
        for dcm_file in data_dir.rglob('*.dcm'):
            total_dicom += 1
        
        # Count processed files
        for png_file in data_dir.rglob('*.png'):
            total_processed += 1
    
    print(f"📊 Total DICOM files: {total_dicom:,}")
    print(f"📊 Total processed files: {total_processed:,}")
    
    if total_dicom > 0:
        print(f"📊 Processing ratio: {total_processed/total_dicom:.1%}")
    
    print("\n🚀 Next Steps:")
    print("-" * 40)
    
    if total_dicom == 0:
        print("1. 📁 Place your DICOM files in notebooks/data/raw/train/ and notebooks/data/raw/test/")
        print("2. 🏗️  Run: ai-health setup")
        print("3. 🔄 Run: ai-health process --data-dir ./notebooks/data")
    elif total_processed == 0:
        print("1. 🔄 Process your DICOM files: ai-health process --data-dir ./notebooks/data")
        print("2. 🎯 Train a model: ai-health train --model-type ResNet50")
    else:
        print("1. 🎯 Train a model: ai-health train --model-type ResNet50")
        print("2. 🖥️  Launch GUI: ai-health gui")
    
    print("3. 📖 Check README.md for detailed instructions")

if __name__ == "__main__":
    check_data_structure()
