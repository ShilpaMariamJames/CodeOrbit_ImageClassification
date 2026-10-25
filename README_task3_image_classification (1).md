# Image Classification Using a Pretrained Model (MobileNetV2)

AI Internship – Task 3

## About
This project classifies a few sample images (dog, bus, cat, banana) using **MobileNetV2**, a deep learning model pretrained on the ImageNet dataset. It shows the top 3 predicted labels with their confidence for each image.

## How the pretrained model works
- MobileNetV2 is a **Convolutional Neural Network (CNN)**. Its early layers detect edges and colours, and its deeper layers detect shapes and whole objects.
- It was **already trained** on about 1.2 million ImageNet images in 1000 categories, so no training is needed.
- Each image is resized to 224×224 and preprocessed. The model outputs a probability for each of the 1000 classes, and the highest ones are the predictions.
- It is small and fast, so it runs without a GPU.

## Tech used
Python, TensorFlow/Keras, NumPy, Pillow, Matplotlib

## How to run
```
pip install tensorflow pillow matplotlib requests
python task3_image_classification.py
```
(Or paste the code into Google Colab and run it.)

## Output
The top-3 predictions are printed for each image, and the image grid with labels is saved as `task3_predictions.png`.
