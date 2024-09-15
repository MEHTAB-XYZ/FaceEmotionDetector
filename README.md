**Face Emotion Detection using FER 2013 Dataset**
This project implements a deep learning model to detect facial emotions using the FER 2013 dataset. The model is capable of recognizing emotions like anger, disgust, fear, happiness, sadness, surprise, and neutrality. Additionally, the project includes a real-time face emotion detection script that utilizes a webcam feed.

**Dataset**
The FER 2013 dataset consists of 35,887 grayscale, 48x48 pixel images of faces, each labeled with one of the seven emotions.

Dataset Source: Kaggle - FER 2013
The dataset is divided into training and testing sets:
Training set: 28,709 images
Testing set: 7,178 images

**Real-Time Emotion Detection**
A Python script, realtimedetection.py, is included to perform real-time face emotion detection using your computer's camera. 
This script leverages OpenCV for face detection and uses the trained model to classify the detected faces into different emotions.
