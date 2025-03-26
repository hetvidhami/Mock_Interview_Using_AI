**Video Analysis for Interview Evaluation**

This program analyzes facial expressions, eye movements, and head poses during video interviews to evaluate candidate engagement and behavior.

**Required Files**

You must have these files in your working directory:

1. Model Files:
   - shape_predictor_68_face_landmarks.dat - dlib's 68-point facial landmark predictor from https://github.com/davisking/dlib-models
   - haarcascade_frontalface_default.xml - OpenCV's Haar Cascade classifier for face detection
   - model.h5 - Pre-trained Keras/TensorFlow model for emotion recognition

2. Python Files:
   - video_analysis.py - Main analysis script

**Installation Instructions**

1. Install required Python packages:
   pip install opencv-python dlib tensorflow scipy numpy matplotlib
