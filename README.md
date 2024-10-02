# **Image Generation Using GAN with CIFAR-10 Dataset**

## **Overview**
This project implements a Generative Adversarial Network (GAN) to generate images using the CIFAR-10 dataset. The model is trained to create synthetic images of categories like cars, airplanes, and animals. It uses TensorFlow for building the model and Streamlit for providing an interactive interface.

## **Features**
- A GAN trained on CIFAR-10 (60,000 images in 10 categories)
- Streamlit-based UI for training the model and generating images
- Ability to generate random images after training

## **Project Structure**
- `app.py`: Script containing the GAN model, training loop, and Streamlit interface.
- `image_gen.py`: Main script containing the CIFAR-10 dataset, GAN model, training loop, and Streamlit interface.
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

## **Comparison with Market-Standard Models**

| **Feature**                  | **Your Model**                  | **Market-Standard Models (e.g., DALL·E, Stable Diffusion, MidJourney)** |
|------------------------------|----------------------------------|------------------------------------------------------------------------|
| **Dataset**                   | CIFAR-10 (60,000 images)         | Hundreds of millions to billions of images (open-domain)               |
| **Image Resolution**          | 32x32 pixels                     | 512x512, 1024x1024, or higher                                          |
| **Training Time**             | ~100,000 epochs (hours/days on GPU) | Weeks or months on large-scale distributed systems                     |
| **Model Architecture**        | Basic GAN with Dense layers      | Advanced architectures (e.g., Diffusion Models, Transformers)          |
| **Complexity of Objects**     | Simple objects (cars, trees, etc.) | Complex and detailed objects, styles, and scenes                       |
| **Number of Parameters**      | Thousands                        | Billions                                                              |
| **Hardware**                  | Basic GPU                        | High-performance distributed GPUs/TPUs                                 |
| **Output Quality**            | Blurry at early stages, improving with more epochs | High-definition, photo-realistic images                                |
| **Customization**             | Limited to noise vector input    | Text-to-image, style transfer, and detailed control                    |
| **Use Case**                  | Learning and experimentation     | Commercial-grade applications (art creation, product design, etc.)     |
| **Deployment**                | Local/Small-Scale                | Cloud-based, large-scale deployments (commercial platforms)            |
