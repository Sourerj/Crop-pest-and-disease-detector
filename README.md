# Crop-pest-and-disease-detector
A Streamlit-based web application that uses a deep learning model to detect crop diseases from leaf images. Upload an image and get instant predictions powered by a custom-trained CNN using transfer learning.

# Crop Disease Detector

This is a web-based crop disease detection system built with Streamlit. The model uses a Convolutional Neural Network (CNN) trained via transfer learning to classify crop leaf images into disease categories.

## Features
- Upload crop leaf images (JPG/PNG)
- Instant prediction of disease or healthy status
- Based on a custom-trained deep learning model (.h5 format)
- Deployed using Streamlit Cloud

## Model Info
- Input shape: 224x224 pixels
- Framework: TensorFlow / Keras
- Model file: `crop_disease_model.h5`
- Example classes: Tomato Bacterial Spot, Early Blight, Healthy, Late Blight

## Setup Instructions
1. Clone the repo and navigate into it.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
