#!/usr/bin/env python3
"""
Main Entry Point for AI Health System
====================================

Command-line interface and main application runner.
"""

import sys
import os
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from . import data_processing, models, utils, reporting, gui

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="AI Health System - Medical Image Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run GUI application
  ai-health gui
  
  # Process dataset
  ai-health process --data-dir ./notebooks/data --target-size 224
  
  # Train model
  ai-health train --model-type ResNet50 --epochs 100
  
  # Generate report
  ai-health report --input results.json --output report.pdf
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # GUI command
    gui_parser = subparsers.add_parser('gui', help='Launch GUI application')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process dataset')
    process_parser.add_argument('--data-dir', default='./notebooks/data', help='Data directory')
    process_parser.add_argument('--target-size', type=int, default=224, help='Target image size')
    process_parser.add_argument('--max-samples', type=int, help='Maximum samples to process')
    process_parser.add_argument('--no-augment', action='store_true', help='Disable data augmentation')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--model-type', default='custom', 
                             choices=['custom', 'ResNet50', 'VGG16', 'EfficientNetB0', 'DenseNet121'],
                             help='Model type to train')
    train_parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate report')
    report_parser.add_argument('--input', required=True, help='Input results file')
    report_parser.add_argument('--output', help='Output report file')
    report_parser.add_argument('--format', choices=['pdf', 'html'], default='pdf', help='Report format')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup project structure')
    setup_parser.add_argument('--base-path', default='.', help='Base path for setup')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'gui':
            run_gui()
        elif args.command == 'process':
            run_data_processing(args)
        elif args.command == 'train':
            run_training(args)
        elif args.command == 'report':
            run_report_generation(args)
        elif args.command == 'setup':
            run_setup(args)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

def run_gui():
    """Run GUI application"""
    print("🚀 Launching AI Health System GUI...")
    gui.main()

def run_data_processing(args):
    """Run data processing pipeline"""
    print("🔄 Starting data processing...")
    
    processor = data_processing.create_processing_pipeline(
        data_dir=args.data_dir,
        target_size=(args.target_size, args.target_size)
    )
    
    # Scan dataset
    class_counts = processor.scan_dataset()
    print(f"📊 Found classes: {class_counts}")
    
    # Process training data
    train_raw = os.path.join(args.data_dir, 'raw', 'train')
    if os.path.exists(train_raw):
        print("🔄 Processing training data...")
        processor.process_dataset(
            source_dir=train_raw,
            output_dir=processor.train_processed,
            max_samples=args.max_samples,
            augment=not args.no_augment
        )
    
    # Process test data
    test_raw = os.path.join(args.data_dir, 'raw', 'test')
    if os.path.exists(test_raw):
        print("🔄 Processing test data...")
        processor.process_dataset(
            source_dir=test_raw,
            output_dir=processor.test_processed,
            max_samples=args.max_samples,
            augment=False
        )
    
    # Create train/validation split
    if os.path.exists(processor.train_processed):
        processor.create_train_val_split(processor.train_processed)
    
    print("✅ Data processing complete!")

def run_training(args):
    """Run model training"""
    print("🎯 Starting model training...")
    
    # Create model
    model = models.create_model(
        model_type=args.model_type,
        input_shape=(224, 224, 3),
        num_classes=2
    )
    
    # Create trainer
    trainer = models.ModelTrainer(model)
    
    print(f"📊 Model created: {args.model_type}")
    print(f"⚙️  Training parameters:")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.learning_rate}")
    
    # Note: This is a placeholder - actual training would require data loading
    print("⚠️  Note: Training requires processed dataset. Use 'process' command first.")
    
    print("✅ Model setup complete!")

def run_report_generation(args):
    """Run report generation"""
    print("📄 Generating report...")
    
    # Load results
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    results = utils.load_results(args.input)
    
    # Generate report
    report_generator = reporting.create_report_generator()
    
    if args.format == 'pdf':
        output_path = report_generator.generate_pdf_report(results, args.output)
    else:
        output_path = report_generator.generate_html_report(results, args.output)
    
    print(f"✅ Report generated: {output_path}")

def run_setup(args):
    """Run project setup"""
    print("🔧 Setting up project structure...")
    
    base_path = Path(args.base_path)
    utils.create_directory_structure(str(base_path))
    
    print("✅ Project setup complete!")

if __name__ == "__main__":
    main()
