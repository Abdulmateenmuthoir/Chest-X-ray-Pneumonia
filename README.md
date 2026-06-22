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

## 🚀 Local Setup Instructions (For Defense & Testing)

Follow these exact steps to pull this project to your local computer and run the Streamlit web app.

### 1. Clone the Repository
Open your terminal (or Command Prompt) and run:
```bash
git clone https://github.com/Abdulmateenmuthoir/Chest-X-ray-Pneumonia.git
cd Chest-X-ray-Pneumonia
```

### 2. Create a Virtual Environment (Recommended)
It is best to run this in an isolated Python environment so it doesn't conflict with other projects.
```bash
# For Mac/Linux:
python3 -m venv venv
source venv/bin/activate

# For Windows:
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
Once your virtual environment is active (you will see `(venv)` in your terminal), install the required packages:
```bash
pip install -r requirements.txt
```

### 4. Run the Web Application Locally
The model weights are already downloaded if you pulled the latest repo. You do not need to re-train the model. Simply launch the web app:
```bash
streamlit run app.py
```
This will open the fully functional dashboard in your browser at `http://localhost:8501`.

---

## 🔬 Developer Commands (Optional)

If you want to re-download the dataset or re-train the model from scratch, use the commands below.

### Download the Dataset
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
