# GAN-model

# Synthetic Chest X-Ray Generation with a DCGAN

This project contains a Deep Convolutional Generative Adversarial Network (DCGAN) implemented in TensorFlow and Keras to generate realistic, synthetic chest X-ray images. The primary goal is to provide a tool for data augmentation and privacy preservation in medical imaging research.

## Overview

Generative Adversarial Networks (GANs) are a class of deep learning models where two neural networks, a **Generator** and a **Discriminator**, compete against each other.
* The **Generator** learns to create plausible data (in this case, X-ray images) from random noise.
* The **Discriminator** learns to distinguish between the generator's fake images and real images from a training dataset.

This adversarial process results in a generator capable of producing high-quality, novel images that mimic the characteristics of the real dataset.

The model is a Deep Convolutional GAN (DCGAN), which uses convolutional and transposed convolutional layers for the discriminator and generator, respectively.

### Generator
The generator takes a 100-dimensional latent noise vector and upsamples it through a series of `Conv2DTranspose` layers to produce a 128x128 grayscale image. `BatchNormalization` is used for stability, and the final activation is `tanh`.


### Discriminator
The discriminator is a standard Convolutional Neural Network (CNN) that takes a 128x128 image and downsamples it, outputting a single probability score indicating whether the image is real or fake. It uses `LeakyReLU` activations and `Dropout` for regularization.

### Prerequisites

We'll need Python 3.8+ and the following libraries:
* TensorFlow
* NumPy
* Matplotlib
