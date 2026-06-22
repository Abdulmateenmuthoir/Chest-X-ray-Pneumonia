"""
Dataset Download Script
Downloads the Chest X-ray Pneumonia dataset from Kaggle using kagglehub.
Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
"""

import os
import kagglehub


def download_dataset():
    """
    Download the chest X-ray pneumonia dataset from Kaggle.
    
    Returns:
        str: Path to the downloaded dataset directory.
    """
    print("=" * 60)
    print("  Downloading Chest X-ray Pneumonia Dataset from Kaggle")
    print("=" * 60)
    
    # Download latest version of the dataset
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    
    print(f"\n✅ Dataset downloaded successfully!")
    print(f"📂 Path to dataset files: {path}")
    
    # Verify directory structure
    dataset_path = os.path.join(path, "chest_xray")
    if not os.path.exists(dataset_path):
        # Sometimes the structure is flat
        dataset_path = path
    
    print(f"\n📁 Dataset directory: {dataset_path}")
    
    # Check for train, val, test directories
    for split in ["train", "val", "test"]:
        split_path = os.path.join(dataset_path, split)
        if os.path.exists(split_path):
            classes = os.listdir(split_path)
            # Filter out hidden files
            classes = [c for c in classes if not c.startswith('.')]
            print(f"\n📂 {split.upper()} set:")
            for cls in sorted(classes):
                cls_path = os.path.join(split_path, cls)
                if os.path.isdir(cls_path):
                    num_images = len([f for f in os.listdir(cls_path) 
                                     if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
                    print(f"   ├── {cls}: {num_images} images")
        else:
            print(f"\n⚠️  {split.upper()} directory not found at: {split_path}")
    
    return dataset_path


if __name__ == "__main__":
    dataset_path = download_dataset()
    print(f"\n{'=' * 60}")
    print(f"  Use this path in your config:")
    print(f"  DATASET_DIR = '{dataset_path}'")
    print(f"{'=' * 60}")
