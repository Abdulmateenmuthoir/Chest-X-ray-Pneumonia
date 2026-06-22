"""
Configuration file for the Pneumonia Classification Project.
Contains all hyperparameters, paths, and settings used across the project.
"""

import os

# ============================================================
# Project Paths
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "model")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

# Dataset paths (will be set after download)
DATASET_DIR = None  # Updated by download_dataset.py

# ============================================================
# Image Configuration
# ============================================================
IMG_HEIGHT = 224        # ResNet50 default input height
IMG_WIDTH = 224         # ResNet50 default input width
IMG_CHANNELS = 3        # RGB channels
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# ============================================================
# Training Hyperparameters
# ============================================================
BATCH_SIZE = 32
EPOCHS_PHASE1 = 10      # Feature extraction phase (frozen base)
EPOCHS_PHASE2 = 10      # Fine-tuning phase (unfrozen top layers)
LEARNING_RATE_PHASE1 = 1e-4
LEARNING_RATE_PHASE2 = 1e-5
VALIDATION_SPLIT = 0.2  # 20% of training data for validation

# ============================================================
# Model Configuration
# ============================================================
DROPOUT_RATE = 0.5
DENSE_UNITS = 256
FINE_TUNE_AT = 140      # Unfreeze layers from this index onwards in ResNet50

# ============================================================
# Callbacks Configuration
# ============================================================
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.2
MIN_LEARNING_RATE = 1e-7

# ============================================================
# Class Labels
# ============================================================
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# ============================================================
# Create output directories
# ============================================================
for directory in [OUTPUT_DIR, MODEL_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)
