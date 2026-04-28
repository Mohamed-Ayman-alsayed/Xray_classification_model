
# #!/usr/bin/env python3
# """
# GUI for Unsupervised Chest X-Ray Anomaly Detection
# =================================================
# - Loads autoencoder model
# - Lets user upload X-ray (png/jpg/dcm)
# - Computes reconstruction error (MSE)
# - Threshold-based anomaly decision
# - Export report as PDF/HTML
# """

# import os
# import sys
# import time
# import numpy as np
# import cv2
# from PIL import Image
# import pydicom
# import tensorflow as tf
# from tensorflow import keras

# from PyQt5.QtWidgets import (
#     QApplication, QMainWindow, QLabel, QPushButton,
#     QFileDialog, QVBoxLayout, QWidget, QMessageBox,
#     QSlider, QProgressBar
# )
# from PyQt5.QtCore import Qt, QThread, pyqtSignal

# # -------------------
# # Config
# # -------------------
# MODEL_PATH = "/home/mohamed/mo-stuff/programing_file/AI-healthSystem/src/unsupervised_2.0.h5"
# IMG_SIZE = (224, 224)

# # -------------------
# # Load Model
# # -------------------
# print(f"📂 Loading model from: {MODEL_PATH}")
# model = keras.models.load_model(MODEL_PATH, compile=False)


# # -------------------
# # Helper: Preprocess
# # -------------------
# def preprocess_image(image_path, target_size=IMG_SIZE):
#     """Load and preprocess image (jpg, png, dcm)."""
#     if image_path.lower().endswith(".dcm"):
#         ds = pydicom.dcmread(image_path)
#         img_array = ds.pixel_array
#         img_array = cv2.resize(img_array, target_size)
#         if len(img_array.shape) == 2:
#             img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
#     else:
#         img = Image.open(image_path).convert("RGB")
#         img = img.resize(target_size)
#         img_array = np.array(img)

#     img_array = img_array.astype(np.float32) / 255.0
#     return np.expand_dims(img_array, axis=0)


# # -------------------
# # Worker Thread
# # -------------------
# class PredictionWorker(QThread):
#     result_ready = pyqtSignal(dict)

#     def __init__(self, image_path, threshold):
#         super().__init__()
#         self.image_path = image_path
#         self.threshold = threshold

#     def run(self):
#         start = time.time()
#         image = preprocess_image(self.image_path)
#         reconstruction = model.predict(image, verbose=0)
#         mse = np.mean(np.square(image - reconstruction))
#         elapsed = time.time() - start

#         diagnosis = "✅ Normal" if mse <= self.threshold else "🚨 Anomaly"

#         result = {
#             "image_path": self.image_path,
#             "mse": float(mse),
#             "threshold": float(self.threshold),
#             "diagnosis": diagnosis,
#             "processing_time": round(elapsed, 2)
#         }
#         self.result_ready.emit(result)


# # -------------------
# # Main GUI
# # -------------------
# class AIHealthSystemGUI(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("AI Health System - Unsupervised X-ray Anomaly Detection")
#         self.setGeometry(200, 200, 600, 400)

#         self.image_path = None
#         self.result = None

#         # Widgets
#         self.label = QLabel("Upload a chest X-ray to analyze", self)
#         self.label.setAlignment(Qt.AlignCenter)

#         self.upload_button = QPushButton("📂 Upload Image")
#         self.upload_button.clicked.connect(self.upload_image)

#         self.predict_button = QPushButton("🔎 Run Analysis")
#         self.predict_button.clicked.connect(self.run_prediction)
#         self.predict_button.setEnabled(False)

#         self.threshold_slider = QSlider(Qt.Horizontal)
#         self.threshold_slider.setRange(1, 50)  # scale 0.001 → 0.05
#         self.threshold_slider.setValue(10)
#         self.threshold_slider.valueChanged.connect(self.update_threshold_label)
#         self.threshold_value = QLabel("Threshold: 0.010")

#         self.progress = QProgressBar()
#         self.progress.setValue(0)

#         self.result_label = QLabel("")
#         self.result_label.setAlignment(Qt.AlignCenter)
#         self.result_label.setStyleSheet("font-size: 16px;")

#         self.report_button = QPushButton("📑 Save Report")
#         self.report_button.clicked.connect(self.save_report)
#         self.report_button.setEnabled(False)

#         # Layout
#         layout = QVBoxLayout()
#         layout.addWidget(self.label)
#         layout.addWidget(self.upload_button)
#         layout.addWidget(self.predict_button)
#         layout.addWidget(self.threshold_value)
#         layout.addWidget(self.threshold_slider)
#         layout.addWidget(self.progress)
#         layout.addWidget(self.result_label)
#         layout.addWidget(self.report_button)

#         container = QWidget()
#         container.setLayout(layout)
#         self.setCentralWidget(container)

#     def update_threshold_label(self):
#         thr = self.threshold_slider.value() / 1000
#         self.threshold_value.setText(f"Threshold: {thr:.3f}")

#     def upload_image(self):
#         file_path, _ = QFileDialog.getOpenFileName(
#             self, "Select Chest X-ray", "", "Images (*.png *.jpg *.jpeg *.dcm)"
#         )
#         if file_path:
#             self.image_path = file_path
#             self.label.setText(f"✅ Loaded: {os.path.basename(file_path)}")
#             self.predict_button.setEnabled(True)

#     def run_prediction(self):
#         if not self.image_path:
#             QMessageBox.warning(self, "Error", "Please upload an image first.")
#             return

#         thr = self.threshold_slider.value() / 1000
#         self.worker = PredictionWorker(self.image_path, thr)
#         self.worker.result_ready.connect(self.display_result)
#         self.progress.setValue(50)
#         self.worker.start()

#     def display_result(self, result):
#         self.result = result
#         self.progress.setValue(100)
#         self.result_label.setText(
#             f"{result['diagnosis']}\nMSE={result['mse']:.6f} (Thr={result['threshold']:.3f})\n"
#             f"Time={result['processing_time']}s"
#         )
#         self.report_button.setEnabled(True)

#     def save_report(self):
#         if not self.result:
#             QMessageBox.warning(self, "Error", "No results to save.")
#             return

#         save_path, _ = QFileDialog.getSaveFileName(
#             self, "Save Report", "", "PDF (*.pdf);;HTML (*.html)"
#         )
#         if save_path:
#             if save_path.endswith(".html"):
#                 with open(save_path, "w") as f:
#                     f.write(f"<h2>AI Health Report</h2>")
#                     f.write(f"<p><b>Image:</b> {self.result['image_path']}</p>")
#                     f.write(f"<p><b>Diagnosis:</b> {self.result['diagnosis']}</p>")
#                     f.write(f"<p><b>MSE:</b> {self.result['mse']:.6f}</p>")
#                     f.write(f"<p><b>Threshold:</b> {self.result['threshold']:.3f}</p>")
#                     f.write(f"<p><b>Processing Time:</b> {self.result['processing_time']}s</p>")
#             else:
#                 from reportlab.lib.pagesizes import A4
#                 from reportlab.pdfgen import canvas
#                 c = canvas.Canvas(save_path, pagesize=A4)
#                 c.setFont("Helvetica", 14)
#                 c.drawString(100, 800, "AI Health System - Report")
#                 c.setFont("Helvetica", 12)
#                 c.drawString(100, 770, f"Image: {self.result['image_path']}")
#                 c.drawString(100, 750, f"Diagnosis: {self.result['diagnosis']}")
#                 c.drawString(100, 730, f"MSE: {self.result['mse']:.6f}")
#                 c.drawString(100, 710, f"Threshold: {self.result['threshold']:.3f}")
#                 c.drawString(100, 690, f"Processing Time: {self.result['processing_time']}s")
#                 c.save()

#             QMessageBox.information(self, "Saved", f"Report saved to {save_path}")


# # -------------------
# # Run App
# # -------------------
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     gui = AIHealthSystemGUI()
#     gui.show()
#     sys.exit(app.exec_())













"""
GUI Module for AI Health System
==============================

PyQt5-based desktop application for chest X-ray analysis with model prediction
and automated report generation.
"""

import sys
import os
import time
from datetime import datetime
from typing import Optional, Dict, Any
import numpy as np
from PIL import Image, ImageQt
import cv2

# Handle imports for both package and standalone execution1
try:
    # Try package imports first
    from .models import create_model, ModelTrainer
    from .data_processing import ChestXRayProcessor
    from .reporting import create_report_generator
    from .utils import plot_confusion_matrix, plot_roc_curve
except ImportError:
    # Fallback for standalone execution
    try:
        from models import create_model, ModelTrainer
        from data_processing import ChestXRayProcessor
        from reporting import create_report_generator
        from utils import plot_confusion_matrix, plot_roc_curve
    except ImportError:
        # If still can't import, add current directory to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        try:
            from models import create_model, ModelTrainer
            from data_processing import ChestXRayProcessor
            from reporting import create_report_generator
            from utils import plot_confusion_matrix, plot_roc_curve
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("Please run this from the project root directory or install the package first.")
            sys.exit(1)

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QTextEdit, QProgressBar, QMessageBox, QFrame,
                             QGroupBox, QGridLayout, QSlider, QSpinBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage, QFont, QPalette, QColor

class PredictionWorker(QThread):
    """Worker thread for model prediction"""
    prediction_complete = pyqtSignal(dict)
    prediction_error = pyqtSignal(str)
    
    def __init__(self, model_path: str, image_path: str):
        super().__init__()
        self.model_path = model_path
        self.image_path = image_path
    
    def run(self):
        try:
            # Load model
         #  trainer = ModelTrainer(None)  # no need to pass model
            trainer.load_model(self.model_path)

            
            # Load and preprocess image
            processor = ChestXRayProcessor()
            image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError("Could not load image")
            
            # Preprocess
            processed_image = processor.preprocess_image(image)
            processed_image = np.expand_dims(processed_image, axis=0)
            
            # Make prediction
            start_time = time.time()
            prediction = trainer.predict(processed_image)
            processing_time = time.time() - start_time
            
            # Get results
            if len(prediction.shape) > 1:
                confidence = float(np.max(prediction))
                predicted_class = int(np.argmax(prediction))
            else:
                confidence = float(prediction[0])
                predicted_class = int(prediction[0] > 0.5)
            
            class_names = ['Normal', 'Pneumonia']
            diagnosis = class_names[predicted_class]
            
            results = {
                'diagnosis': diagnosis,
                'confidence': confidence,
                'processing_time': processing_time,
                'model_name': os.path.basename(self.model_path),
                'image_size': f"{image.shape[1]}x{image.shape[0]}",
                'patient_id': os.path.basename(self.image_path).split('.')[0],
                'study_date': datetime.now().strftime("%Y-%m-%d")
            }
            
            self.prediction_complete.emit(results)
            
        except Exception as e:
            self.prediction_error.emit(str(e))

class AIHealthSystemGUI(QMainWindow):
    """Main GUI window for AI Health System"""
    
    def __init__(self):
        super().__init__()
        self.model_path = None
        self.current_image_path = None
        self.prediction_results = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("AI Health System - Chest X-Ray Analysis")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 10px 20px;
                text-align: center;
                font-size: 14px;
                border-radius: 5px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QLabel {
                font-size: 12px;
            }
            QTextEdit {
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 5px;
                font-family: 'Courier New';
                font-size: 11px;
            }
        """)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Image and controls
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Right panel - Results and reports
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 1)
        
        # Status bar
        self.statusBar().showMessage("Ready - Please load a model and image")
        
    def create_left_panel(self) -> QWidget:
        """Create left panel with image display and controls"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Model selection group
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout(model_group)
        
        self.model_label = QLabel("No model loaded")
        self.model_label.setStyleSheet("color: #666666; font-style: italic;")
        
        load_model_btn = QPushButton("Load Model")
        load_model_btn.clicked.connect(self.load_model)
        
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(load_model_btn)
        layout.addWidget(model_group)
        
        # Image upload group
        image_group = QGroupBox("Image Upload")
        image_layout = QVBoxLayout(image_group)
        
        upload_btn = QPushButton("Upload X-Ray Image")
        upload_btn.clicked.connect(self.upload_image)
        
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #cccccc;
                border-radius: 5px;
                background-color: #fafafa;
                color: #666666;
            }
        """)
        
        image_layout.addWidget(upload_btn)
        image_layout.addWidget(self.image_label)
        layout.addWidget(image_group)
        
        # Analysis group
        analysis_group = QGroupBox("Analysis")
        analysis_layout = QVBoxLayout(analysis_group)
        
        self.analyze_btn = QPushButton("Analyze Image")
        self.analyze_btn.clicked.connect(self.analyze_image)
        self.analyze_btn.setEnabled(False)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        analysis_layout.addWidget(self.analyze_btn)
        analysis_layout.addWidget(self.progress_bar)
        layout.addWidget(analysis_group)
        
        layout.addStretch()
        return panel
    
    def create_right_panel(self) -> QWidget:
        """Create right panel with results and report generation"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Results group
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(200)
        self.results_text.setPlaceholderText("Results will appear here after analysis...")
        
        results_layout.addWidget(self.results_text)
        layout.addWidget(results_group)
        
        # Report generation group
        report_group = QGroupBox("Report Generation")
        report_layout = QVBoxLayout(report_group)
        
        self.pdf_report_btn = QPushButton("Generate PDF Report")
        self.pdf_report_btn.clicked.connect(self.generate_pdf_report)
        self.pdf_report_btn.setEnabled(False)
        
        self.html_report_btn = QPushButton("Generate HTML Report")
        self.html_report_btn.clicked.connect(self.generate_html_report)
        self.html_report_btn.setEnabled(False)
        
        report_layout.addWidget(self.pdf_report_btn)
        report_layout.addWidget(self.html_report_btn)
        layout.addWidget(report_group)
        
        # Settings group
        settings_group = QGroupBox("Settings")
        settings_layout = QGridLayout(settings_group)
        
        settings_layout.addWidget(QLabel("Confidence Threshold:"), 0, 0)
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(50, 95)
        self.confidence_slider.setValue(70)
        self.confidence_slider.setTickPosition(QSlider.TicksBelow)
        self.confidence_slider.setTickInterval(5)
        
        self.confidence_label = QLabel("70%")
        self.confidence_slider.valueChanged.connect(
            lambda v: self.confidence_label.setText(f"{v}%")
        )
        
        settings_layout.addWidget(self.confidence_slider, 0, 1)
        settings_layout.addWidget(self.confidence_label, 0, 2)
        
        layout.addWidget(settings_group)
        
        layout.addStretch()
        return panel
    
    def load_model(self):
         file_path, _ = QFileDialog.getOpenFileName(
        self, "Select Model File", "", "Model Files (*.h5 *.hdf5);;All Files (*)"
        )
        if file_path:
            try:
                from tensorflow import keras
                keras.models.load_model(file_path, compile=False)  # validate it loads
               
                self.model_path = file_path
                self.model_label.setText(f"Model: {os.path.basename(file_path)}")
                self.model_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
                self.statusBar().showMessage(f"Model loaded: {os.path.basename(file_path)}")
                self.check_ready_state()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
    
    def upload_image(self):
        """Upload an X-ray image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select X-Ray Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.dcm);;All Files (*)"
        )
        
        if file_path:
            try:
                self.current_image_path = file_path
                
                # Display image
                if file_path.lower().endswith('.dcm'):
                    # Handle DICOM files
                    import pydicom
                    dicom_data = pydicom.dcmread(file_path)
                    image_array = dicom_data.pixel_array
                    
                    # Convert to PIL Image
                    image = Image.fromarray(image_array)
                else:
                    # Handle regular image files
                    image = Image.open(file_path)
                
                # Convert to QPixmap and display - use multiple fallback methods
                pixmap = None
                
                # Method 1: Try ImageQt conversion
                try:
                    qimage = ImageQt.ImageQt(image)
                    pixmap = QPixmap.fromImage(qimage)
                except Exception as e:
                    print(f"ImageQt conversion failed: {e}")
                
                # Method 2: Try numpy array conversion
                if pixmap is None:
                    try:
                        # Convert PIL image to numpy array
                        if image.mode == 'RGBA':
                            image = image.convert('RGB')
                        
                        # Convert to numpy array
                        img_array = np.array(image)
                        
                        # Convert to QImage
                        height, width, channel = img_array.shape
                        bytes_per_line = 3 * width
                        qimage = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
                        pixmap = QPixmap.fromImage(qimage)
                    except Exception as e:
                        print(f"Numpy conversion failed: {e}")
                
                # Method 3: Try OpenCV conversion
                if pixmap is None:
                    try:
                        # Use OpenCV to load and convert
                        cv_image = cv2.imread(file_path)
                        if cv_image is not None:
                            # Convert BGR to RGB
                            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                            
                            # Convert to QImage
                            height, width, channel = cv_image.shape
                            bytes_per_line = 3 * width
                            qimage = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
                            pixmap = QPixmap.fromImage(qimage)
                    except Exception as e:
                        print(f"OpenCV conversion failed: {e}")
                
                if pixmap is None:
                    raise ValueError("All image conversion methods failed")
                
                # Scale to fit label while maintaining aspect ratio
                scaled_pixmap = pixmap.scaled(
                    self.image_label.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                
                self.image_label.setPixmap(scaled_pixmap)
                self.image_label.setStyleSheet("border: none; background-color: transparent;")
                
                self.statusBar().showMessage(f"Image loaded: {os.path.basename(file_path)}")
                self.check_ready_state()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
                print(f"Image loading error details: {e}")
                import traceback
                traceback.print_exc()
    
    def check_ready_state(self):
        """Check if ready to analyze"""
        ready = self.model_path is not None and self.current_image_path is not None
        self.analyze_btn.setEnabled(ready)
        
        if ready:
            self.statusBar().showMessage("Ready to analyze - Click 'Analyze Image' to start")
    
    def analyze_image(self):
        """Analyze the uploaded image"""
        if not self.model_path or not self.current_image_path:
            return
        
        # Disable buttons during analysis
        self.analyze_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        # Start prediction worker
        self.prediction_worker = PredictionWorker(self.model_path, self.current_image_path)
        self.prediction_worker.prediction_complete.connect(self.on_prediction_complete)
        self.prediction_worker.prediction_error.connect(self.on_prediction_error)
        self.prediction_worker.start()
        
        self.statusBar().showMessage("Analyzing image...")
    
    def on_prediction_complete(self, results: Dict[str, Any]):
        """Handle prediction completion"""
        self.prediction_results = results
        
        # Update results display
        results_text = f"""ANALYSIS RESULTS
{'='*50}

Diagnosis: {results['diagnosis']}
Confidence: {results['confidence']:.1%}
Processing Time: {results['processing_time']:.2f} seconds

Model: {results['model_name']}
Image Size: {results['image_size']}
Patient ID: {results['patient_id']}
Study Date: {results['study_date']}

Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        self.results_text.setText(results_text)
        
        # Enable report generation
        self.pdf_report_btn.setEnabled(True)
        self.html_report_btn.setEnabled(True)
        
        # Reset UI
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # Check confidence threshold
        confidence_threshold = self.confidence_slider.value() / 100.0
        if results['confidence'] < confidence_threshold:
            self.statusBar().showMessage(
                f"Analysis complete - Low confidence ({results['confidence']:.1%} < {confidence_threshold:.1%})"
            )
        else:
            self.statusBar().showMessage("Analysis complete - High confidence result")
    
    def on_prediction_error(self, error_msg: str):
        """Handle prediction error"""
        QMessageBox.critical(self, "Analysis Error", f"Failed to analyze image: {error_msg}")
        
        # Reset UI
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Analysis failed")
    
    def generate_pdf_report(self):
        """Generate PDF report"""
        if not self.prediction_results:
            return
        
        try:
            report_generator = create_report_generator()
            output_path = report_generator.generate_pdf_report(self.prediction_results)
            
            QMessageBox.information(
                self, "Success", 
                f"PDF report generated successfully!\nSaved to: {output_path}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate PDF report: {str(e)}")
    
    def generate_html_report(self):
        """Generate HTML report"""
        if not self.prediction_results:
            return
        
        try:
            report_generator = create_report_generator()
            output_path = report_generator.generate_html_report(self.prediction_results)
            
            QMessageBox.information(
                self, "Success", 
                f"HTML report generated successfully!\nSaved to: {output_path}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate HTML report: {str(e)}")
    
    def closeEvent(self, event):
        """Handle application close event"""
        if hasattr(self, 'prediction_worker') and self.prediction_worker.isRunning():
            self.prediction_worker.terminate()
            self.prediction_worker.wait()
        event.accept()

def main():
    """Main function to run the GUI application"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("AI Health System")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("AI Health System Team")
    
    # Create and show main window
    window = AIHealthSystemGUI()
    window.show()
    
    # Run application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
