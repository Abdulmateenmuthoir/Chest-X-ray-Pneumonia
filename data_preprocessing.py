"""
Data Preprocessing Module
Handles data loading, augmentation, and generator creation for training,
validation, and testing datasets.
"""

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import config


def get_data_generators(dataset_dir):
    """
    Create data generators for training, validation, and testing.
    
    The Kaggle dataset has a very small validation set (16 images),
    so we use 20% of the training data as validation instead.
    
    Args:
        dataset_dir (str): Path to the dataset root directory.
        
    Returns:
        tuple: (train_generator, val_generator, test_generator)
    """
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")
    
    print("\n📊 Setting up data generators...")
    print(f"   Training directory: {train_dir}")
    print(f"   Test directory:     {test_dir}")
    
    # --------------------------------------------------------
    # Training Data Generator (with augmentation)
    # --------------------------------------------------------
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,            # Normalize pixel values to [0, 1]
        rotation_range=20,               # Random rotation up to 20 degrees
        width_shift_range=0.1,           # Horizontal shift up to 10%
        height_shift_range=0.1,          # Vertical shift up to 10%
        shear_range=0.1,                 # Shear transformation
        zoom_range=0.15,                 # Random zoom up to 15%
        horizontal_flip=True,            # Random horizontal flip
        fill_mode="nearest",             # Fill strategy for new pixels
        validation_split=config.VALIDATION_SPLIT  # 20% for validation
    )
    
    # --------------------------------------------------------
    # Test Data Generator (no augmentation, only rescaling)
    # --------------------------------------------------------
    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0
    )
    
    # --------------------------------------------------------
    # Create generators
    # --------------------------------------------------------
    
    # Training generator (80% of training data)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="binary",             # Binary classification
        color_mode="rgb",
        shuffle=True,
        seed=42,
        subset="training"               # Use training subset
    )
    
    # Validation generator (20% of training data)
    val_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="binary",
        color_mode="rgb",
        shuffle=False,
        seed=42,
        subset="validation"             # Use validation subset
    )
    
    # Test generator
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="binary",
        color_mode="rgb",
        shuffle=False
    )
    
    # Print dataset information
    print(f"\n📈 Dataset Summary:")
    print(f"   Training samples:   {train_generator.samples}")
    print(f"   Validation samples: {val_generator.samples}")
    print(f"   Test samples:       {test_generator.samples}")
    print(f"   Class indices:      {train_generator.class_indices}")
    print(f"   Image size:         {config.IMG_SIZE}")
    print(f"   Batch size:         {config.BATCH_SIZE}")
    
    return train_generator, val_generator, test_generator


def compute_class_weights(train_generator):
    """
    Compute class weights to handle class imbalance.
    The dataset has more pneumonia images than normal images.
    
    Args:
        train_generator: Training data generator.
        
    Returns:
        dict: Class weights dictionary.
    """
    # Get all labels from the generator
    labels = train_generator.classes
    
    # Compute balanced class weights
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )
    
    class_weight_dict = dict(enumerate(class_weights))
    
    print(f"\n⚖️  Class Weights (to handle imbalance):")
    for cls_idx, weight in class_weight_dict.items():
        cls_name = list(train_generator.class_indices.keys())[cls_idx]
        count = np.sum(labels == cls_idx)
        print(f"   {cls_name}: {weight:.4f} ({count} samples)")
    
    return class_weight_dict


if __name__ == "__main__":
    # Test the data generators
    # You need to run download_dataset.py first
    from download_dataset import download_dataset
    
    dataset_dir = download_dataset()
    train_gen, val_gen, test_gen = get_data_generators(dataset_dir)
    class_weights = compute_class_weights(train_gen)
    
    # Display a sample batch
    print(f"\n🔍 Sample batch shape: {next(iter(train_gen))[0].shape}")
