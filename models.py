"""
Models Module for AI Health System
=================================

Contains CNN architectures for chest X-ray analysis, including custom models
and transfer learning with pre-trained architectures.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
import numpy as np
from typing import Tuple, Dict, Any, Optional, Union
import os


class ChestXRayCNN:
    """Custom CNN architecture for chest X-ray classification"""
    
    def __init__(self, input_shape: Tuple[int, int, int] = (224, 224, 3), 
                 num_classes: int = 2, dropout_rate: float = 0.5):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.model = None
        
    def build_model(self) -> keras.Model:

        def create_resnet50(input_shape=(224,224,3), num_classes=2, trainable_layers=50):
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
            # Freeze most layers
            for layer in base_model.layers[:-trainable_layers]:
                layer.trainable = False
            for layer in base_model.layers[-trainable_layers:]:
                layer.trainable = True

            x = layers.GlobalAveragePooling2D()(base_model.output)
            x = layers.Dense(256, activation='relu')(x)
            x = layers.Dropout(0.4)(x)
            outputs = layers.Dense(num_classes, activation='softmax')(x)

            model = models.Model(inputs=base_model.input, outputs=outputs)
            return model

        self.model = create_resnet50(input_shape=self.input_shape, num_classes=self.num_classes)
        return self.model
    
    def compile_model(self, learning_rate: float = 0.001) -> None:
        """Compile the model with optimizer and loss"""
        if self.model is None:
            self.build_model()
        
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        loss = 'categorical_crossentropy' if self.num_classes > 2 else 'binary_crossentropy'
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
        )
    
    def get_callbacks(self, model_save_path: str = 'best_model.h5') -> list:
        """Get training callbacks"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                model_save_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        return callbacks


class TransferLearningModel:
    """Transfer learning with pre-trained models"""
    
    def __init__(self, base_model_name: str = 'ResNet50', 
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 2, dropout_rate: float = 0.5):
        self.base_model_name = base_model_name
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.model = None
        self.base_model = None
        
    def get_base_model(self) -> keras.Model:
        """Get pre-trained base model"""
        if self.base_model_name == 'ResNet50':
            base_model = applications.ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.base_model_name == 'VGG16':
            base_model = applications.VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.base_model_name == 'EfficientNetB0':
            base_model = applications.EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.base_model_name == 'DenseNet121':
            base_model = applications.DenseNet121(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Unsupported base model: {self.base_model_name}")
        
        self.base_model = base_model
        return base_model
    
    def build_model(self, fine_tune_layers: int = 0) -> keras.Model:
        """Build transfer learning model"""
        base_model = self.get_base_model()
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Create the model
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate * 0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate * 0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Fine-tune last few layers if specified
        if fine_tune_layers > 0:
            base_model.trainable = True
            for layer in base_model.layers[:-fine_tune_layers]:
                layer.trainable = False
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate: float = 0.001) -> None:
        """Compile the model"""
        if self.model is None:
            self.build_model()
        
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        loss = 'categorical_crossentropy' if self.num_classes > 2 else 'binary_crossentropy'
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
        )
    
    def get_callbacks(self, model_save_path: str = 'best_transfer_model.h5') -> list:
        """Get training callbacks"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                model_save_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        return callbacks


class ModelTrainer:
    """Training wrapper for models"""
    
    def __init__(self, model: Union[ChestXRayCNN, TransferLearningModel, keras.Model]):
        self.model = model
        self.history = None
    
    def _get_keras_model(self):
        """Always return the underlying keras.Model"""
        if isinstance(self.model, (ChestXRayCNN, TransferLearningModel)):
            return self.model.model
        return self.model
    
    def train(self, train_data: np.ndarray, train_labels: np.ndarray,
              validation_data: Tuple[np.ndarray, np.ndarray] = None,
              epochs: int = 100, batch_size: int = 32,
              model_save_path: str = None) -> keras.callbacks.History:
        """Train the model"""
        keras_model = self._get_keras_model()
        if keras_model is None:
            self.model.compile_model()
            keras_model = self._get_keras_model()
        
        # Get callbacks
        if model_save_path is None:
            model_save_path = f"best_{type(self.model).__name__.lower()}.h5"
        
        callbacks = self.model.get_callbacks(model_save_path) if hasattr(self.model, "get_callbacks") else []
        
        # Train
        self.history = keras_model.fit(
            train_data, train_labels,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, test_data: np.ndarray, test_labels: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        keras_model = self._get_keras_model()
        if keras_model is None:
            raise ValueError("Model not trained yet")
        
        results = keras_model.evaluate(test_data, test_labels, verbose=0)
        
        metrics = {}
        if hasattr(keras_model, 'metrics_names'):
            for i, metric_name in enumerate(keras_model.metrics_names):
                metrics[metric_name] = float(results[i])
        else:
            metrics['loss'] = float(results[0])
            metrics['accuracy'] = float(results[1])
        
        return metrics
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Make predictions"""
        keras_model = self._get_keras_model()
        if keras_model is None:
            raise ValueError("Model not trained yet")
        return keras_model.predict(data)
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model"""
        keras_model = self._get_keras_model()
        if keras_model is not None:
            keras_model.save(filepath)
            print(f"✅ Model saved to: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model"""
        if os.path.exists(filepath):
            from tensorflow.keras.losses import MeanSquaredError

            self.model = keras.models.load_model(
                filepath,
                custom_objects={"mse": MeanSquaredError()}
            )

            print(f"✅ Model loaded from: {filepath}")
        else:
            raise FileNotFoundError(f"Model file not found: {filepath}")


def create_model(model_type, input_shape=(224,224,3), num_classes=2):
    """Create model with the specified architecture"""
    if model_type == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

        # Freeze all except last 50 layers
        for layer in base_model.layers[:-50]:
            layer.trainable = False
        for layer in base_model.layers[-50:]:
            layer.trainable = True

        x = layers.GlobalAveragePooling2D()(base_model.output)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)

        model = models.Model(inputs=base_model.input, outputs=outputs)
        return model
    else:
        raise ValueError(f"Unknown model_type: {model_type}")







# """

# Models Module for AI Health System
# =================================

# Contains CNN architectures for chest X-ray analysis, including custom models
# and transfer learning with pre-trained architectures.
# """

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers, applications, optimizers
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras import layers, models
# import numpy as np
# from typing import Tuple, Dict, Any, Optional, Union
# import os

# class ChestXRayCNN:
#     """Custom CNN architecture for chest X-ray classification"""
    
#     def __init__(self, input_shape: Tuple[int, int, int] = (224, 224, 3), 
#                  num_classes: int = 2, dropout_rate: float = 0.5):
#         self.input_shape = input_shape
#         self.num_classes = num_classes
#         self.dropout_rate = dropout_rate
#         self.model = None
        
#     def build_model(self) -> keras.Model:

#         def create_resnet50(input_shape=(224,224,3), num_classes=2, trainable_layers=50):
#             base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
#     # Freeze most layers
#             for layer in base_model.layers[:-trainable_layers]:
#                         layer.trainable = False
#             for layer in base_model.layers[-trainable_layers:]:
#                         layer.trainable = True

#             x = layers.GlobalAveragePooling2D()(base_model.output)
#             x = layers.Dense(256, activation='relu')(x)
#             x = layers.Dropout(0.4)(x)
#             outputs = layers.Dense(num_classes, activation='softmax')(x)

#             model = models.Model(inputs=base_model.input, outputs=outputs)
#             return model

#         def create_model(model_type, input_shape, num_classes):
#             if model_type == "ResNet50":
#                 return create_resnet50(input_shape=input_shape, num_classes=num_classes)
#             else:
#                 raise ValueError(f"Unknown model_type: {model_type}")
#         # """Build custom CNN architecture"""
#         # model = keras.Sequential([
#         #     # Input layer
#         #     layers.Input(shape=self.input_shape),
            
#         #     # First convolutional block
#         #     layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
#         #     layers.BatchNormalization(),
#         #     layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
#         #     layers.BatchNormalization(),
#         #     layers.MaxPooling2D((2, 2)),
#         #     layers.Dropout(self.dropout_rate * 0.5),
            
#         #     # Second convolutional block
#         #     layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#         #     layers.BatchNormalization(),
#         #     layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#         #     layers.BatchNormalization(),
#         #     layers.MaxPooling2D((2, 2)),
#         #     layers.Dropout(self.dropout_rate * 0.5),
            
#         #     # Third convolutional block
#         #     layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
#         #     layers.BatchNormalization(),
#         #     layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
#         #     layers.BatchNormalization(),
#         #     layers.MaxPooling2D((2, 2)),
#         #     layers.Dropout(self.dropout_rate * 0.5),
            
#         #     # Fourth convolutional block
#         #     layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
#         #     layers.BatchNormalization(),
#         #     layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
#         #     layers.BatchNormalization(),
#         #     layers.MaxPooling2D((2, 2)),
#         #     layers.Dropout(self.dropout_rate * 0.5),
            
#         #     # Global pooling and dense layers
#         #     layers.GlobalAveragePooling2D(),
#         #     layers.Dense(512, activation='relu'),
#         #     layers.BatchNormalization(),
#         #     layers.Dropout(self.dropout_rate),
#         #     layers.Dense(256, activation='relu'),
#         #     layers.BatchNormalization(),
#         #     layers.Dropout(self.dropout_rate * 0.5),
#         #     layers.Dense(self.num_classes, activation='softmax')
#         # ])
        
#         # self.model = model
#         # return model
    
#     def compile_model(self, learning_rate: float = 0.001) -> None:
#         """Compile the model with optimizer and loss"""
#         if self.model is None:
#             self.build_model()
        
#         optimizer = optimizers.Adam(learning_rate=learning_rate)
#         loss = 'categorical_crossentropy' if self.num_classes > 2 else 'binary_crossentropy'
        
#         self.model.compile(
#             optimizer=optimizer,
#             loss=loss,
#             metrics=['accuracy', 'precision', 'recall', 'AUC']
#         )
    
#     def get_callbacks(self, model_save_path: str = 'best_model.h5') -> list:
#         """Get training callbacks"""
#         callbacks = [
#             EarlyStopping(
#                 monitor='val_loss',
#                 patience=15,
#                 restore_best_weights=True,
#                 verbose=1
#             ),
#             ReduceLROnPlateau(
#                 monitor='val_loss',
#                 factor=0.5,
#                 patience=8,
#                 min_lr=1e-7,
#                 verbose=1
#             ),
#             ModelCheckpoint(
#                 model_save_path,
#                 monitor='val_loss',
#                 save_best_only=True,
#                 verbose=1
#             )
#         ]
#         return callbacks

# class TransferLearningModel:
#     """Transfer learning with pre-trained models"""
    
#     def __init__(self, base_model_name: str = 'ResNet50', 
#                  input_shape: Tuple[int, int, int] = (224, 224, 3),
#                  num_classes: int = 2, dropout_rate: float = 0.5):
#         self.base_model_name = base_model_name
#         self.input_shape = input_shape
#         self.num_classes = num_classes
#         self.dropout_rate = dropout_rate
#         self.model = None
#         self.base_model = None
        
#     def get_base_model(self) -> keras.Model:
#         """Get pre-trained base model"""
#         if self.base_model_name == 'ResNet50':
#             base_model = applications.ResNet50(
#                 weights='imagenet',
#                 include_top=False,
#                 input_shape=self.input_shape
#             )
#         elif self.base_model_name == 'VGG16':
#             base_model = applications.VGG16(
#                 weights='imagenet',
#                 include_top=False,
#                 input_shape=self.input_shape
#             )
#         elif self.base_model_name == 'EfficientNetB0':
#             base_model = applications.EfficientNetB0(
#                 weights='imagenet',
#                 include_top=False,
#                 input_shape=self.input_shape
#             )
#         elif self.base_model_name == 'DenseNet121':
#             base_model = applications.DenseNet121(
#                 weights='imagenet',
#                 include_top=False,
#                 input_shape=self.input_shape
#             )
#         else:
#             raise ValueError(f"Unsupported base model: {self.base_model_name}")
        
#         self.base_model = base_model
#         return base_model
    
#     def build_model(self, fine_tune_layers: int = 0) -> keras.Model:
#         """Build transfer learning model"""
#         base_model = self.get_base_model()
        
#         # Freeze base model layers
#         base_model.trainable = False
        
#         # Create the model
#         model = keras.Sequential([
#             base_model,
#             layers.GlobalAveragePooling2D(),
#             layers.BatchNormalization(),
#             layers.Dropout(self.dropout_rate),
#             layers.Dense(512, activation='relu'),
#             layers.BatchNormalization(),
#             layers.Dropout(self.dropout_rate * 0.5),
#             layers.Dense(256, activation='relu'),
#             layers.BatchNormalization(),
#             layers.Dropout(self.dropout_rate * 0.5),
#             layers.Dense(self.num_classes, activation='softmax')
#         ])
        
#         # Fine-tune last few layers if specified
#         if fine_tune_layers > 0:
#             base_model.trainable = True
#             for layer in base_model.layers[:-fine_tune_layers]:
#                 layer.trainable = False
        
#         self.model = model
#         return model
    
#     def compile_model(self, learning_rate: float = 0.001) -> None:
#         """Compile the model"""
#         if self.model is None:
#             self.build_model()
        
#         optimizer = optimizers.Adam(learning_rate=learning_rate)
#         loss = 'categorical_crossentropy' if self.num_classes > 2 else 'binary_crossentropy'
        
#         self.model.compile(
#             optimizer=optimizer,
#             loss=loss,
#             metrics=['accuracy', 'precision', 'recall', 'AUC']
#         )
    
#     def get_callbacks(self, model_save_path: str = 'best_transfer_model.h5') -> list:
#         """Get training callbacks"""
#         callbacks = [
#             EarlyStopping(
#                 monitor='val_loss',
#                 patience=20,
#                 restore_best_weights=True,
#                 verbose=1
#             ),
#             ReduceLROnPlateau(
#                 monitor='val_loss',
#                 factor=0.2,
#                 patience=10,
#                 min_lr=1e-7,
#                 verbose=1
#             ),
#             ModelCheckpoint(
#                 model_save_path,
#                 monitor='val_loss',
#                 save_best_only=True,
#                 verbose=1
#             )
#         ]
#         return callbacks

# class ModelTrainer:
#     """Training wrapper for models"""
    
#     def __init__(self, model: Union[ChestXRayCNN, TransferLearningModel]):
#         self.model = model
#         self.history = None
    
#     def train(self, train_data: np.ndarray, train_labels: np.ndarray,
#               validation_data: Tuple[np.ndarray, np.ndarray] = None,
#               epochs: int = 100, batch_size: int = 32,
#               model_save_path: str = None) -> keras.callbacks.History:
#         """Train the model"""
#         if not hasattr(self.model, 'model') or self.model.model is None:
#             self.model.compile_model()
        
#         # Get callbacks
#         if model_save_path is None:
#             model_save_path = f"best_{type(self.model).__name__.lower()}.h5"
        
#         callbacks = self.model.get_callbacks(model_save_path)
        
#         # Train
#         self.history = self.model.model.fit(
#             train_data, train_labels,
#             validation_data=validation_data,
#             epochs=epochs,
#             batch_size=batch_size,
#             callbacks=callbacks,
#             verbose=1
#         )
        
#         return self.history
    
#     def evaluate(self, test_data: np.ndarray, test_labels: np.ndarray) -> Dict[str, float]:
#         """Evaluate model performance"""
#         if self.model.model is None:
#             raise ValueError("Model not trained yet")
        
#         results = self.model.model.evaluate(test_data, test_labels, verbose=0)
        
#         metrics = {}
#         if hasattr(self.model.model, 'metrics_names'):
#             for i, metric_name in enumerate(self.model.model.metrics_names):
#                 metrics[metric_name] = float(results[i])
#         else:
#             metrics['loss'] = float(results[0])
#             metrics['accuracy'] = float(results[1])
        
#         return metrics
    
#     def predict(self, data: np.ndarray) -> np.ndarray:
#         """Make predictions"""
#         if self.model.model is None:
#             raise ValueError("Model not trained yet")
#         return self.model.predict(data)
    
#     def save_model(self, filepath: str) -> None:
#         """Save the trained model"""
#         if self.model.model is not None:
#             self.model.model.save(filepath)
#             print(f"✅ Model saved to: {filepath}")
    
#     def load_model(self, filepath: str) -> None:
#         """Load a trained model"""
#         if os.path.exists(filepath):
#             from tensorflow.keras.losses import MeanSquaredError

#             self.model = keras.models.load_model(
#                 filepath,
#                 custom_objects={"mse": MeanSquaredError()}
#             )

#             print(f"✅ Model loaded from: {filepath}")
#         else:
#             raise FileNotFoundError(f"Model file not found: {filepath}")

# def create_model(model_type, input_shape=(224,224,3), num_classes=2):
#     """Create model with the specified architecture"""
#     if model_type == 'ResNet50':
#         base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

#         # Freeze all except last 50 layers
#         for layer in base_model.layers[:-50]:
#             layer.trainable = False
#         for layer in base_model.layers[-50:]:
#             layer.trainable = True

#         x = layers.GlobalAveragePooling2D()(base_model.output)
#         x = layers.Dense(256, activation='relu')(x)
#         x = layers.Dropout(0.4)(x)
#         outputs = layers.Dense(num_classes, activation='softmax')(x)

#         model = models.Model(inputs=base_model.input, outputs=outputs)
#         return model
#     else:
#         raise ValueError(f"Unknown model_type: {model_type}")
