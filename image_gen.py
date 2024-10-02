import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Build Generator
def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, activation='relu', input_dim=100))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(32 * 32 * 3, activation='tanh'))  # 32x32 pixels, 3 channels (RGB)
    model.add(layers.Reshape((32, 32, 3)))  # Output shape (32, 32, 3)
    return model

# Build Discriminator
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Flatten(input_shape=(32, 32, 3)))  # Input shape (32, 32, 3)
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary output: real (1) or fake (0)
    return model

# Build GAN
def build_gan(generator, discriminator):
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    discriminator.trainable = False
    gan_input = layers.Input(shape=(100,))
    generated_image = generator(gan_input)
    gan_output = discriminator(generated_image)
    gan = tf.keras.Model(gan_input, gan_output)
    gan.compile(optimizer='adam', loss='binary_crossentropy')
    return gan

# Train the GAN
def train_gan(generator, discriminator, gan, epochs=10000, batch_size=128):
    (X_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()  # Load CIFAR-10 dataset
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5  # Normalize to [-1, 1]

    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_images = X_train[idx]

        noise = np.random.normal(0, 1, (batch_size, 100))
        fake_images = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        if epoch % 1000 == 0:
            st.write(f"{epoch} [D loss: {d_loss[0]}] [G loss: {g_loss}]")
            plot_generated_images(generator, epoch)

# Plot Generated Images
def plot_generated_images(generator, epoch, examples=10):
    noise = np.random.normal(0, 1, (examples, 100))
    generated_images = generator.predict(noise)
    generated_images = (generated_images + 1) / 2.0  # Rescale from [-1, 1] to [0, 1]

    fig, axes = plt.subplots(1, examples, figsize=(20, 2))
    for i in range(examples):
        axes[i].imshow(generated_images[i])
        axes[i].axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    st.image(img, caption=f'Generated Images at Epoch {epoch}', use_column_width=True)

# Streamlit Interface
def main():
    st.title("Image Generation using GANs (CIFAR-10 Dataset)")
    
    st.write("This is an image generation AI using Generative Adversarial Networks (GANs) trained on the CIFAR-10 dataset.")

    if st.button('Train GAN'):
        generator = build_generator()
        discriminator = build_discriminator()
        gan = build_gan(generator, discriminator)
        st.write("Training GAN... This will take some time.")
        train_gan(generator, discriminator, gan, epochs=10000, batch_size=128)
    
    if st.button('Generate Images'):
        generator = build_generator()
        plot_generated_images(generator, epoch=0)

if __name__ == "__main__":
    main()
