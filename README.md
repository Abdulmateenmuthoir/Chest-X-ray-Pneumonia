# Automated Pneumonia Classification from Chest Radiographs Using Deep CNNs

> **Student:** Dauda Suliyat Adewumi (AI/HND/F24/0073)  
> **Supervisor:** Mr. Amari B.R  
> **Institution:** Department of Artificial Intelligence, School of Computing, Federal Polytechnic Offa

## 📌 Overview

This project implements an automated pneumonia classification system using **ResNet50** (a deep Convolutional Neural Network) with **transfer learning**. The system analyzes chest X-ray images and classifies them as either **Normal** or **Pneumonia**.

### Key Features
- Deep learning model based on ResNet50 with transfer learning
- Two-phase training: feature extraction + fine-tuning
- Data augmentation for improved generalization
- Class weight balancing for handling imbalanced data
- Comprehensive evaluation metrics and visualizations
- Interactive web application for live demonstrations

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | TensorFlow / Keras |
| Model | ResNet50 (pre-trained on ImageNet) |
| Dataset | Kaggle Chest X-ray Pneumonia Dataset |
| Web App | Streamlit |
| Visualization | Matplotlib, Seaborn |

## 📁 Project Structure

```
Chest_X-ray/
├── requirements.txt          # Dependencies
├── README.md                 # Project documentation (this file)
├── config.py                 # Configuration and hyperparameters
├── download_dataset.py       # Dataset download script
├── data_preprocessing.py     # Data loading and augmentation
├── model.py                  # ResNet50 model architecture
├── train.py                  # Training pipeline
├── evaluate.py               # Model evaluation and visualization
├── predict.py                # Single image prediction
├── app.py                    # Streamlit web application
└── outputs/
    ├── model/                # Saved model weights
    ├── plots/                # Evaluation plots
    └── logs/                 # Training logs
```

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the Dataset

```bash
python download_dataset.py
```

> **Note:** This requires Kaggle API credentials. Place your `kaggle.json` in `~/.kaggle/`.

### 3. Train the Model

```bash
python train.py
```

Training consists of two phases:
- **Phase 1:** Feature extraction with frozen ResNet50 base (10 epochs)
- **Phase 2:** Fine-tuning with unfrozen top layers (10 epochs)

### 4. Evaluate the Model

```bash
python evaluate.py
```

This generates:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Training/Validation Curves
- Sample Predictions

### 5. Predict on a Single Image

```bash
python predict.py --image path/to/chest_xray.jpeg
```

### 6. Run the Web Application

```bash
streamlit run app.py
```

## 📊 Model Architecture

```
ResNet50 (Pre-trained on ImageNet)
    └── GlobalAveragePooling2D
        └── Dense(256, ReLU)
            └── BatchNormalization
                └── Dropout(0.5)
                    └── Dense(1, Sigmoid) → Output
```

## 📈 Evaluation Metrics

After training, the model is evaluated using:
- **Accuracy** — Overall correct predictions
- **Precision** — True positive rate among positive predictions
- **Recall (Sensitivity)** — True positive rate among actual positives
- **F1-Score** — Harmonic mean of precision and recall
- **Confusion Matrix** — Visual breakdown of predictions

## 📚 References

1. Tang, Y. X., et al. (2020). Automated abnormality classification of chest radiographs using deep CNNs.
2. Lakhani, P., & Sundaram, B. (2017). Deep learning at chest radiography.
3. Rahman, T., et al. (2020). Transfer learning for pneumonia detection.
4. He, K., et al. (2016). Deep residual learning for image recognition (ResNet).

## ⚠️ Disclaimer

This system is developed for educational and research purposes only. It should NOT be used as a substitute for professional medical diagnosis.

## 📄 License

This project is submitted as part of the HND final year project requirements at Federal Polytechnic Offa.
