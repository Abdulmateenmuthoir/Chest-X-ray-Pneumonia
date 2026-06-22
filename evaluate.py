"""
Evaluation Script
Performs detailed evaluation of the trained model on the test set.
Generates metrics, confusion matrix, and training/validation curves.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from tensorflow.keras.models import load_model
import config
from download_dataset import download_dataset
from data_preprocessing import get_data_generators


def plot_training_history(history, save_dir):
    """
    Plot training and validation accuracy/loss curves.
    
    Args:
        history (dict): Training history dictionary.
        save_dir (str): Directory to save plots.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(history["accuracy"], label="Training Accuracy", linewidth=2, color="#2196F3")
    axes[0].plot(history["val_accuracy"], label="Validation Accuracy", linewidth=2, color="#FF9800")
    axes[0].set_title("Model Accuracy", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # Loss plot
    axes[1].plot(history["loss"], label="Training Loss", linewidth=2, color="#2196F3")
    axes[1].plot(history["val_loss"], label="Validation Loss", linewidth=2, color="#FF9800")
    axes[1].set_title("Model Loss", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📈 Training curves saved to: {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_dir):
    """
    Plot and save confusion matrix as a heatmap.
    
    Args:
        y_true (array): True labels.
        y_pred (array): Predicted labels.
        save_dir (str): Directory to save the plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        linewidths=1,
        linecolor="white",
        annot_kws={"size": 16}
    )
    plt.title("Confusion Matrix", fontsize=16, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 Confusion matrix saved to: {save_path}")
    
    return cm


def plot_sample_predictions(model, test_generator, save_dir, num_samples=12):
    """
    Plot a grid of sample predictions with actual vs predicted labels.
    
    Args:
        model: Trained Keras model.
        test_generator: Test data generator.
        save_dir (str): Directory to save the plot.
        num_samples (int): Number of samples to display.
    """
    # Get a batch of test images
    test_generator.reset()
    images, labels = next(iter(test_generator))
    
    # Make predictions
    predictions = model.predict(images, verbose=0)
    pred_classes = (predictions > 0.5).astype(int).flatten()
    
    # Plot grid
    cols = 4
    rows = min(num_samples // cols, 3)
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
    
    for idx, ax in enumerate(axes.flatten()):
        if idx >= len(images):
            ax.axis("off")
            continue
        
        ax.imshow(images[idx])
        
        true_label = config.CLASS_NAMES[int(labels[idx])]
        pred_label = config.CLASS_NAMES[int(pred_classes[idx])]
        confidence = predictions[idx][0] if pred_classes[idx] == 1 else 1 - predictions[idx][0]
        
        color = "green" if true_label == pred_label else "red"
        ax.set_title(
            f"True: {true_label}\nPred: {pred_label} ({confidence:.1%})",
            fontsize=10,
            color=color,
            fontweight="bold"
        )
        ax.axis("off")
    
    plt.suptitle("Sample Predictions (Green=Correct, Red=Incorrect)", 
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "sample_predictions.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   🖼️  Sample predictions saved to: {save_path}")


def evaluate():
    """
    Run full evaluation pipeline on the test set.
    """
    print("=" * 60)
    print("  AUTOMATED PNEUMONIA CLASSIFICATION")
    print("  Model Evaluation")
    print("=" * 60)
    
    # --------------------------------------------------------
    # Step 1: Load the trained model
    # --------------------------------------------------------
    model_path = os.path.join(config.MODEL_DIR, "pneumonia_classifier_final.keras")
    
    if not os.path.exists(model_path):
        print(f"\n❌ Error: Model not found at {model_path}")
        print("   Please run 'python train.py' first to train the model.")
        return
    
    print(f"\n📂 Loading model from: {model_path}")
    model = load_model(model_path)
    print("✅ Model loaded successfully!")
    
    # --------------------------------------------------------
    # Step 2: Load test data
    # --------------------------------------------------------
    print("\n📥 Loading test data...")
    dataset_dir = download_dataset()
    _, _, test_gen = get_data_generators(dataset_dir)
    
    # --------------------------------------------------------
    # Step 3: Generate predictions
    # --------------------------------------------------------
    print("\n🔮 Generating predictions on test set...")
    test_gen.reset()
    
    predictions = model.predict(test_gen, verbose=1)
    pred_classes = (predictions > 0.5).astype(int).flatten()
    true_classes = test_gen.classes
    
    # --------------------------------------------------------
    # Step 4: Calculate metrics
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    
    accuracy = accuracy_score(true_classes, pred_classes)
    precision = precision_score(true_classes, pred_classes)
    recall = recall_score(true_classes, pred_classes)
    f1 = f1_score(true_classes, pred_classes)
    
    print(f"\n   📊 Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"   📊 Precision: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"   📊 Recall:    {recall:.4f} ({recall * 100:.2f}%)")
    print(f"   📊 F1-Score:  {f1:.4f} ({f1 * 100:.2f}%)")
    
    # Detailed classification report
    print(f"\n{'=' * 60}")
    print("  CLASSIFICATION REPORT")
    print(f"{'=' * 60}")
    report = classification_report(
        true_classes, pred_classes,
        target_names=config.CLASS_NAMES
    )
    print(report)
    
    # Save metrics to JSON
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "total_test_samples": int(len(true_classes)),
        "correct_predictions": int(np.sum(pred_classes == true_classes)),
        "incorrect_predictions": int(np.sum(pred_classes != true_classes))
    }
    
    metrics_path = os.path.join(config.LOGS_DIR, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n💾 Metrics saved to: {metrics_path}")
    
    # --------------------------------------------------------
    # Step 5: Generate plots
    # --------------------------------------------------------
    print(f"\n📊 Generating visualization plots...")
    
    # Confusion Matrix
    cm = plot_confusion_matrix(true_classes, pred_classes, config.PLOTS_DIR)
    
    # Training curves (if history exists)
    history_path = os.path.join(config.LOGS_DIR, "training_history.json")
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
        plot_training_history(history, config.PLOTS_DIR)
    else:
        print("   ⚠️  Training history not found, skipping training curves")
    
    # Sample predictions
    plot_sample_predictions(model, test_gen, config.PLOTS_DIR)
    
    print(f"\n{'=' * 60}")
    print(f"  Evaluation Complete! 🎉")
    print(f"  All plots saved to: {config.PLOTS_DIR}")
    print(f"{'=' * 60}")
    
    return metrics


if __name__ == "__main__":
    evaluate()
