# **Image Generation Using GAN with CIFAR-10 Dataset**

## **Overview**
This project implements a Generative Adversarial Network (GAN) to generate images using the CIFAR-10 dataset. The model is trained to create synthetic images of categories like cars, airplanes, and animals. It uses TensorFlow for building the model and Streamlit for providing an interactive interface.

## **Features**
- A GAN trained on CIFAR-10 (60,000 images in 10 categories)
- Streamlit-based UI for training the model and generating images
- Ability to generate random images after training

## **Project Structure**
- `app.py`: Main script containing the GAN model, training loop, and Streamlit interface.
- `requirements.txt`: List of dependencies to run the project.
- `README.md`: Project documentation.
  
## **Dataset**
- **CIFAR-10**: The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes:
  - Airplanes
  - Automobiles
  - Birds
  - Cats
  - Deer
  - Dogs
  - Frogs
  - Horses
  - Ships
  - Trucks
