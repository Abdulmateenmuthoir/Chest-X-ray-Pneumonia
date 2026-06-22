"""
Single Image Prediction Utility
Loads a trained model and predicts whether a chest X-ray shows
pneumonia or is normal.
"""

import os
import sys
import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import config


def predict_image(image_path, model_path=None):
    """
    Predict whether a chest X-ray image shows pneumonia or is normal.
    
    Args:
        image_path (str): Path to the chest X-ray image.
        model_path (str): Path to the trained model. If None, uses default.
        
    Returns:
        dict: Prediction result with class name and confidence.
    """
    # Validate image path
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at {image_path}")
        return None
    
    # Load model
    if model_path is None:
        model_path = os.path.join(config.MODEL_DIR, "pneumonia_classifier_final.keras")
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        print("   Please run 'python train.py' first to train the model.")
        return None
    
    print(f"📂 Loading model from: {model_path}")
    model = load_model(model_path)
    
    # Load and preprocess image
    print(f"🖼️  Loading image: {image_path}")
    img = load_img(image_path, target_size=config.IMG_SIZE, color_mode="rgb")
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(img_array, verbose=0)[0][0]
    
    # Interpret result
    predicted_class = 1 if prediction > 0.5 else 0
    class_name = config.CLASS_NAMES[predicted_class]
    confidence = prediction if predicted_class == 1 else 1 - prediction
    
    # Display result
    print("\n" + "=" * 50)
    print("  PREDICTION RESULT")
    print("=" * 50)
    
    if class_name == "PNEUMONIA":
        print(f"  ⚠️  Diagnosis: {class_name}")
    else:
        print(f"  ✅ Diagnosis: {class_name}")
    
    print(f"  📊 Confidence: {confidence:.2%}")
    print(f"  📊 Raw Score:  {prediction:.6f}")
    print("=" * 50)
    
    result = {
        "image_path": image_path,
        "predicted_class": class_name,
        "confidence": float(confidence),
        "raw_score": float(prediction)
    }
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict pneumonia from a chest X-ray image"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the chest X-ray image"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to the trained model (optional)"
    )
    
    args = parser.parse_args()
    result = predict_image(args.image, args.model)
