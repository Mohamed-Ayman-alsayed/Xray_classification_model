#!/usr/bin/env python3
"""
GUI Launcher for AI Health System
=================================

Simple script to launch the GUI application from the project root.
"""

import sys
import os

# Determine the project root and add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = current_dir  # this file already lives inside the src directory
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import and run GUI
try:
    from gui import main
    print("🚀 Launching AI Health System GUI...")
    main()
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\n💡 Solutions:")
    print("1. Make sure you're in the project root directory")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Install package: pip install -e .")
    print("4. Or use: ai-health gui")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error launching GUI: {e}")
    sys.exit(1)
