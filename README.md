😷 Face Mask Detection using CNN

An AI-powered face mask detection system built using a Convolutional Neural Network (CNN) with TensorFlow/Keras.
The model classifies whether a person is wearing a mask or not from an image.

The project also includes a web application built with Streamlit that allows users to:
Upload an image
Capture an image using a webcam
Get real-time predictions from the trained model

📌 Project Overview
Face mask detection became an important computer vision task during the COVID-19 pandemic.
This project uses deep learning and CNN architecture to detect whether a person is wearing a mask.

The trained model takes an image of size 128 × 128 × 3 and predicts one of two classes:
Mask
No Mask

🧠 Model Architecture
The CNN model was implemented using TensorFlow / Keras.

Model Layers
Conv2D (32 filters, 3×3, ReLU)
MaxPooling2D (2×2)

Conv2D (64 filters, 3×3, ReLU)
MaxPooling2D (2×2)

Flatten

Dense (128, ReLU)
Dropout (0.25)

Dense (64, ReLU)
Dropout (0.25)

Dense (2, Sigmoid)
Input Shape
(128, 128, 3)

📊 Model Performance
Model evaluation on the test dataset:
Metric	Value
Accuracy	92.38%
Loss	0.21

⚙️ Training Configuration
Optimizer:
Adam
Loss Function:
Sparse Categorical Crossentropy
Callbacks used:
EarlyStopping
Early stopping configuration:
monitor = 'val_loss'
patience = 3
restore_best_weights = True

🚀 Web Application Features
The Streamlit application provides an interactive interface for predictions.

Features:
Upload an image for mask detection
Capture image using webcam
Real-time CNN prediction
Displays prediction confidence

Simple and user-friendly UI

🖥️ Demo Workflow
User Uploads Image / Uses Webcam
            ↓
Image Preprocessing
(Resize → Normalize → Expand dimensions)
            ↓
CNN Model Prediction
            ↓
Output:
Mask / No Mask

📂 Project Structure
Mask_Detection
│
├── app.py
├── model.keras
├── requirements.txt
└── README.md

🛠️ Technologies Used
Python
TensorFlow / Keras
Convolutional Neural Networks (CNN)
OpenCV
NumPy
Pillow
Streamlit
