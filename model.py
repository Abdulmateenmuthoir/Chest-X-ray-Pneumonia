"""
Model Architecture Module
Defines the ResNet50-based CNN model for pneumonia classification.
Uses transfer learning with a pre-trained ResNet50 backbone.
"""

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.optimizers import Adam
import config


def build_model(fine_tune=False):
    """
    Build the ResNet50-based classification model.
    
    Architecture:
        - ResNet50 base (pre-trained on ImageNet, no top)
        - GlobalAveragePooling2D
        - Dense(256, relu)
        - BatchNormalization
        - Dropout(0.5)
        - Dense(1, sigmoid) — binary output
    
    Args:
        fine_tune (bool): If True, unfreeze the top layers of ResNet50
                         for fine-tuning. Default is False (feature extraction).
    
    Returns:
        tensorflow.keras.Model: Compiled model ready for training.
    """
    print("\n🏗️  Building ResNet50 Model...")
    
    # --------------------------------------------------------
    # Load pre-trained ResNet50 (without the classification head)
    # --------------------------------------------------------
    base_model = ResNet50(
        weights="imagenet",              # Pre-trained on ImageNet
        include_top=False,               # Remove the original classification head
        input_shape=config.INPUT_SHAPE   # (224, 224, 3)
    )
    
    # --------------------------------------------------------
    # Freeze or unfreeze base model layers
    # --------------------------------------------------------
    if fine_tune:
        # Unfreeze layers from FINE_TUNE_AT onwards for fine-tuning
        for layer in base_model.layers[:config.FINE_TUNE_AT]:
            layer.trainable = False
        for layer in base_model.layers[config.FINE_TUNE_AT:]:
            layer.trainable = True
        
        trainable = sum(1 for layer in base_model.layers if layer.trainable)
        frozen = sum(1 for layer in base_model.layers if not layer.trainable)
        print(f"   Fine-tuning mode: {trainable} trainable, {frozen} frozen layers")
    else:
        # Freeze all base model layers (feature extraction only)
        base_model.trainable = False
        print(f"   Feature extraction mode: All {len(base_model.layers)} base layers frozen")
    
    # --------------------------------------------------------
    # Add custom classification head
    # --------------------------------------------------------
    x = base_model.output
    x = GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = Dense(config.DENSE_UNITS, activation="relu", name="dense_256")(x)
    x = BatchNormalization(name="batch_norm")(x)
    x = Dropout(config.DROPOUT_RATE, name="dropout")(x)
    predictions = Dense(1, activation="sigmoid", name="output")(x)
    
    # --------------------------------------------------------
    # Create the full model
    # --------------------------------------------------------
    model = Model(inputs=base_model.input, outputs=predictions, name="PneumoniaClassifier")
    
    # --------------------------------------------------------
    # Compile the model
    # --------------------------------------------------------
    learning_rate = config.LEARNING_RATE_PHASE2 if fine_tune else config.LEARNING_RATE_PHASE1
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    # Print model summary
    total_params = model.count_params()
    trainable_params = sum([p.numpy().size for p in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    print(f"\n📋 Model Summary:")
    print(f"   Total parameters:         {total_params:,}")
    print(f"   Trainable parameters:     {trainable_params:,}")
    print(f"   Non-trainable parameters: {non_trainable_params:,}")
    print(f"   Learning rate:            {learning_rate}")
    print(f"   Optimizer:                Adam")
    print(f"   Loss function:            Binary Crossentropy")
    
    return model


def load_trained_model(model_path):
    """
    Load a previously trained model from disk.
    
    Args:
        model_path (str): Path to the saved model file (.keras or .h5).
        
    Returns:
        tensorflow.keras.Model: Loaded model.
    """
    from tensorflow.keras.models import load_model
    
    print(f"\n📂 Loading model from: {model_path}")
    model = load_model(model_path)
    print("✅ Model loaded successfully!")
    
    return model


if __name__ == "__main__":
    # Test model building
    print("Phase 1: Feature Extraction")
    model = build_model(fine_tune=False)
    model.summary()
    
    print("\n" + "=" * 60)
    
    print("\nPhase 2: Fine-Tuning")
    model_ft = build_model(fine_tune=True)
