# Anime Face Generator using DCGAN 

This repository contains a **Deep Learning–based Anime Face Generator** developed as a **Semester project**, focused on generating anime-style character faces using **Deep Convolutional Generative Adversarial Networks (DCGANs)**.

The project demonstrates how GANs can learn complex visual patterns—such as **hair styles, hair colors, eye structures, and facial shapes**—from a **limited anime face dataset** and synthesize new, realistic-looking faces through adversarial training.

---

##  Project Overview

- Implemented a **DCGAN architecture from scratch** using **TensorFlow / Keras**
- Generator synthesizes **anime-style RGB face images**
- Discriminator classifies **real vs generated images**
- Trained on a dataset of **~1700 anime character images**
- Target image resolution: **64 × 64 × 3**
- Implemented and trained using **Google Colab (GPU)**

This project reflects **hands-on experimentation with GANs**, CNN-based image synthesis, and custom deep learning training loops.

---

##  Model Architecture

### Generator
- **Input:** 100-dimensional latent noise vector  
- Dense projection → reshape to **8 × 8 × 512**
- Series of **Conv2DTranspose** layers for upsampling:
  - 8×8 → 16×16 → 32×32 → 64×64
- **ReLU** activations
- Final **tanh** activation for image generation
- **~6 million trainable parameters**
- Weight initialization: `RandomNormal(mean=0, stddev=0.02)` (DCGAN standard)

### Discriminator
- **Input:** 64 × 64 × 3 RGB image
- Convolutional layers with stride-based downsampling
- **Batch Normalization + LeakyReLU (α = 0.2)**
- **Dropout** for regularization
- Final **sigmoid** output (real vs fake)
- **~400k trainable parameters**

### GAN Type
- **Deep Convolutional GAN (DCGAN)**

---

##  Training Methodology

- Custom **DCGAN model subclass** using `keras.Model`
- Manual implementation of the **`train_step`** function
- Separate optimizers for Generator and Discriminator
- **Binary Cross-Entropy loss**
- **Label smoothing** applied to real samples
- Adversarial training with alternating updates:
  - Discriminator learns to distinguish real vs fake
  - Generator learns to fool the discriminator

Training was performed on **GPU using Google Colab** due to the computational cost of GAN training.

---

##  Output Generation & Monitoring

- Custom **Keras callback** used to:
  - Periodically generate image grids during training
  - Visually inspect generator performance across epochs
- Images generated from **random latent noise vectors**
- Output images correctly normalized and denormalized for visualization

> **Note:** Training outputs and generated images were cleared to keep the repository lightweight.  
> The notebook preserves the **full architecture, training logic, and image generation pipeline**.

---

##  Repository Contents

├── DCGAN_Anime_Generator.ipynb
└── README.md


The notebook includes:
- Dataset loading and normalization
- Generator and Discriminator definitions
- Custom DCGAN training loop
- Training monitoring logic
- Sample image generation from random noise

---

##  Key Learning Outcomes

- Practical understanding of **GAN training dynamics**
- Handling **generator–discriminator imbalance**
- Exposure to **mode collapse** and instability issues
- Deep experience with **CNN-based image synthesis**
- Implementation of **custom training loops** in TensorFlow/Keras

---

##  Limitations

- Relatively **small dataset** (~1700 images)
- GAN training is **computationally expensive**
- Generated outputs cleared to reduce repository size
- No advanced GAN variants implemented

---

##  Future Improvements

- Conditional GAN for **explicit attribute control**
- Higher-resolution image generation
- Implementation of **WGAN-GP** or **StyleGAN**
- Deployment as a **web demo** using Flask or Streamlit

---

##  Authors

- **Taifoor Asrar**  
- **Muhammad Usman Iftikhar**  
  *BS Artificial Intelligence*

