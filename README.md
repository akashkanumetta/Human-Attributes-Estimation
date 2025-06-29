## Human Height and Weight Estimation from Selfie Images

This project predicts a person’s height, weight, age, and gender from a single front-facing image using deep learning. Designed to work through a web interface or mobile device, the system estimates physical attributes from facial and upper-body features captured via a selfie camera.

Project Overview

A transfer learning approach is used with MobileNetV2 as the feature extractor. The model is trained on a labeled dataset where attributes are embedded in filenames. The model outputs continuous values for height, weight, and age, and a binary classification for gender. Data augmentation is applied to enhance model generalization.

Features

* End-to-end prediction of height, weight, gender, and age
* Uses MobileNetV2 for lightweight feature extraction
* Supports selfie or portrait images with minimal preprocessing
* Custom loss functions and metrics for multi-output regression and classification
* Trained model saved in HDF5 format for deployment

Dataset

The dataset includes labeled images of individuals with filenames encoding the attributes:

* Height in feet
* Weight in kilograms
* Gender (male/female)
* Age in years

Example filename: `1000_5.7h_70w_male_46a.jpg`

All images are resized to 224x224 and normalized. Data is split into training and test sets using an 80:20 ratio.

Technologies Used

* Python
* TensorFlow and Keras for model training
* NumPy for data manipulation
* Regular expressions for label extraction
* MobileNetV2 for transfer learning

Use Cases

* Health and fitness apps for biometric estimation
* Demographic data collection in surveys
* Virtual try-on or body measurement tools
* Identity verification systems based on visual attributes

Future Improvements

* Use of face/body landmarks for better spatial context
* Integration with a web-based or mobile front end
* Real-time prediction from camera feed
* Fine-tuning with larger and more diverse datasets

