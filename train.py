"""
Training Script
Handles the two-phase training process:
    Phase 1: Feature extraction with frozen ResNet50 base
    Phase 2: Fine-tuning with unfrozen top layers
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger
)
import config
from download_dataset import download_dataset
from data_preprocessing import get_data_generators, compute_class_weights
from model import build_model


def get_callbacks(phase):
    """
    Create training callbacks for a given phase.
    
    Args:
        phase (str): Either 'phase1' or 'phase2'.
        
    Returns:
        list: List of Keras callbacks.
    """
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        # Save the best model based on validation loss
        ModelCheckpoint(
            filepath=os.path.join(config.MODEL_DIR, f"best_model_{phase}.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        # Reduce learning rate when validation loss plateaus
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=config.MIN_LEARNING_RATE,
            verbose=1
        ),
        # Log training metrics to CSV
        CSVLogger(
            os.path.join(config.LOGS_DIR, f"training_log_{phase}.csv"),
            append=False
        )
    ]
    
    return callbacks


def train():
    """
    Execute the full two-phase training pipeline.
    
    Phase 1: Feature Extraction
        - Freeze all ResNet50 base layers
        - Train only the custom classification head
        - Higher learning rate (1e-4)
        
    Phase 2: Fine-Tuning
        - Unfreeze top layers of ResNet50
        - Train with lower learning rate (1e-5)
        - Continue from Phase 1 weights
    """
    print("=" * 60)
    print("  AUTOMATED PNEUMONIA CLASSIFICATION")
    print("  Training Pipeline")
    print("=" * 60)
    
    # --------------------------------------------------------
    # Step 1: Download/locate dataset
    # --------------------------------------------------------
    print("\n📥 Step 1: Loading Dataset...")
    dataset_dir = download_dataset()
    
    # --------------------------------------------------------
    # Step 2: Create data generators
    # --------------------------------------------------------
    print("\n📊 Step 2: Creating Data Generators...")
    train_gen, val_gen, test_gen = get_data_generators(dataset_dir)
    
    # --------------------------------------------------------
    # Step 3: Compute class weights
    # --------------------------------------------------------
    print("\n⚖️  Step 3: Computing Class Weights...")
    class_weights = compute_class_weights(train_gen)
    
    # --------------------------------------------------------
    # Step 4: Phase 1 — Feature Extraction
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("  PHASE 1: Feature Extraction (Frozen Base)")
    print("=" * 60)
    
    model = build_model(fine_tune=False)
    
    history_phase1 = model.fit(
        train_gen,
        epochs=config.EPOCHS_PHASE1,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=get_callbacks("phase1"),
        verbose=1
    )
    
    print("\n✅ Phase 1 training complete!")
    
    # --------------------------------------------------------
    # Step 5: Phase 2 — Fine-Tuning
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("  PHASE 2: Fine-Tuning (Unfrozen Top Layers)")
    print("=" * 60)
    
    # Rebuild model with fine-tuning enabled
    model = build_model(fine_tune=True)
    
    # Load Phase 1 weights
    phase1_model_path = os.path.join(config.MODEL_DIR, "best_model_phase1.keras")
    if os.path.exists(phase1_model_path):
        model.load_weights(phase1_model_path)
        print("✅ Loaded Phase 1 weights for fine-tuning")
    
    # Recompile with lower learning rate
    from tensorflow.keras.optimizers import Adam
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE_PHASE2),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    history_phase2 = model.fit(
        train_gen,
        epochs=config.EPOCHS_PHASE2,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=get_callbacks("phase2"),
        verbose=1
    )
    
    print("\n✅ Phase 2 training complete!")
    
    # --------------------------------------------------------
    # Step 6: Save final model
    # --------------------------------------------------------
    final_model_path = os.path.join(config.MODEL_DIR, "pneumonia_classifier_final.keras")
    model.save(final_model_path)
    print(f"\n💾 Final model saved to: {final_model_path}")
    
    # --------------------------------------------------------
    # Step 7: Save training history
    # --------------------------------------------------------
    # Combine histories
    combined_history = {}
    for key in history_phase1.history:
        combined_history[key] = history_phase1.history[key] + history_phase2.history[key]
    
    history_path = os.path.join(config.LOGS_DIR, "training_history.json")
    # Convert numpy values to Python floats for JSON serialization
    serializable_history = {
        key: [float(v) for v in values]
        for key, values in combined_history.items()
    }
    with open(history_path, "w") as f:
        json.dump(serializable_history, f, indent=2)
    
    print(f"📊 Training history saved to: {history_path}")
    
    # --------------------------------------------------------
    # Step 8: Quick evaluation on test set
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Quick Evaluation on Test Set")
    print("=" * 60)
    
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=1)
    print(f"\n📊 Test Results:")
    print(f"   Test Loss:     {test_loss:.4f}")
    print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    
    print("\n" + "=" * 60)
    print("  Training Complete! 🎉")
    print(f"  Run 'python evaluate.py' for detailed evaluation")
    print("=" * 60)
    
    return model, combined_history


if __name__ == "__main__":
    # Set memory growth to avoid GPU memory issues
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"🖥️  Using GPU: {gpus}")
    else:
        print("🖥️  No GPU detected, using CPU")
    
    model, history = train()
