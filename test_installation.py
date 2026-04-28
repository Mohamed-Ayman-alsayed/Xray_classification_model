#!/usr/bin/env python3
"""
Test script to verify AI Health System installation
"""

import sys
import os

def test_imports():
    """Test if all modules can be imported"""
    print("🧪 Testing module imports...")
    
    try:
        # Test core imports
        import numpy as np
        print("✅ NumPy imported successfully")
        
        import pandas as pd
        print("✅ Pandas imported successfully")
        
        import tensorflow as tf
        print("✅ TensorFlow imported successfully")
        
        import cv2
        print("✅ OpenCV imported successfully")
        
        import pydicom
        print("✅ PyDICOM imported successfully")
        
        import matplotlib.pyplot as plt
        print("✅ Matplotlib imported successfully")
        
        import seaborn as sns
        print("✅ Seaborn imported successfully")
        
        import PyQt5
        print("✅ PyQt5 imported successfully")
        
        import reportlab
        print("✅ ReportLab imported successfully")
        
        import jinja2
        print("✅ Jinja2 imported successfully")
        
        print("\n🎉 All core dependencies imported successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_src_imports():
    """Test if source modules can be imported"""
    print("\n🧪 Testing source module imports...")
    
    try:
        # Add src to path
        src_path = os.path.join(os.path.dirname(__file__), 'src')
        sys.path.insert(0, src_path)
        
        # Test source imports
        from data_processing import ChestXRayProcessor
        print("✅ Data processing module imported successfully")
        
        from models import create_model, ModelTrainer
        print("✅ Models module imported successfully")
        
        from utils import plot_training_history, create_directory_structure
        print("✅ Utils module imported successfully")
        
        from reporting import create_report_generator
        print("✅ Reporting module imported successfully")
        
        from gui import AIHealthSystemGUI
        print("✅ GUI module imported successfully")
        
        print("\n🎉 All source modules imported successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Source import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test directory creation
        from utils import create_directory_structure
        create_directory_structure('./test_output')
        print("✅ Directory creation works")
        
        # Test model creation
        from models import create_model
        model = create_model('custom', num_classes=2)
        print("✅ Model creation works")
        
        # Test processor creation with correct path
        from data_processing import create_processing_pipeline
        processor = create_processing_pipeline('./notebooks/data')
        print("✅ Processor creation works")
        
        # Cleanup test directory
        import shutil
        if os.path.exists('./test_output'):
            shutil.rmtree('./test_output')
        print("✅ Cleanup completed")
        
        print("\n🎉 Basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False

def test_data_directory():
    """Test if data directory exists and has expected structure"""
    print("\n🧪 Testing data directory structure...")
    
    try:
        data_dir = './notebooks/data'
        if os.path.exists(data_dir):
            print(f"✅ Data directory found: {data_dir}")
            
            # Check subdirectories
            expected_dirs = ['raw', 'processed', 'reports']
            for subdir in expected_dirs:
                subdir_path = os.path.join(data_dir, subdir)
                if os.path.exists(subdir_path):
                    print(f"✅ Found subdirectory: {subdir}")
                else:
                    print(f"⚠️  Missing subdirectory: {subdir}")
            
            # Check raw data
            raw_dir = os.path.join(data_dir, 'raw')
            if os.path.exists(raw_dir):
                raw_contents = os.listdir(raw_dir)
                print(f"✅ Raw data directory contains: {raw_contents}")
            else:
                print("⚠️  Raw data directory not found")
                
        else:
            print(f"⚠️  Data directory not found: {data_dir}")
            print("   This is expected if you haven't set up the data structure yet.")
        
        return True
        
    except Exception as e:
        print(f"❌ Data directory test error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 AI Health System - Installation Test")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test source modules
    src_ok = test_src_imports()
    
    # Test functionality
    func_ok = test_basic_functionality()
    
    # Test data directory
    data_ok = test_data_directory()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    if imports_ok and src_ok and func_ok:
        print("🎉 ALL TESTS PASSED!")
        print("✅ AI Health System is ready to use!")
        print("\n🚀 Next steps:")
        print("   1. Run: ai-health setup")
        print("   2. Run: ai-health gui")
        print("   3. Check the README.md for usage instructions")
        print("\n📁 Note: Data directory is located at: ./notebooks/data")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the error messages above and fix any issues.")
        print("\n💡 Common solutions:")
        print("   1. Install missing dependencies: pip install -r requirements.txt")
        print("   2. Check Python version (3.8+ required)")
        print("   3. Verify all files are in the correct locations")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
