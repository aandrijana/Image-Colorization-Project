# Image-Colorization-Project
Students: Andrijana Kešelj, Anđela Maksimović

This is the final version of the image colorization project done by students of DSAI ETF Sarajevo.

Important note: Due to inability to delete past commits and some errors present in our previous GitHub repository's files, we decided to push the whole project only after completing everything.


# 🎨 Image Colorization

This project uses deep learning to automatically colorize grayscale images, restoring realistic and visually coherent colors.
## Table of Contents
- [👀 Overview](#-overview)
- [➡️ Introduction](#️-introduction)
- [📦 Usage](#-usage)
- [🖌️ Color Space Choice](#️-color-space-choice)
- [🗃️ Dataset](#️-dataset)
- [🧹 Data Preparation](#️-data-preparation)
- [⌨️ Project Challenges and Key Concepts](#️-project-challenges-and-key-concepts)
- [⚙️ Model version and optimizers](#️-model-version-and-optimizers)
- [👀 Results](#-results)
- [📏 Metrics](#-metrics)
- [🏁 Conclusion and future steps](#-conclusion-and-future-steps)
- [📜 References](#-references)
- [✒️ License](#-license)
  

## 👀 Overview

The main goal of this project is to develop an AI model capable of transforming grayscale images into their colorized counterparts. The aim is to model the complex relationship between luminance (grayscale intensity) and plausible color distributions using deep learning techniques. By learning from large datasets of color images, the model can predict realistic and visually appealing color versions of black-and-white inputs.

This project showcases the power of AI in creative image processing tasks and explores techniques such as convolutional neural networks (CNNs), image-to-image translation, and loss function optimization to achieve high-quality colorization results.


## ➡️Introduction

This project tackles the classic and visually striking deep learning challenge of image colorization—converting grayscale images into plausible, colorful ones. Unlike simple pixel mappings, colorization requires the model to understand the semantic content of images, learning that skies tend to be blue, grass is green, and context matters.

We implemented and experimented with state-of-the-art architectures like U-Net and GANs, gaining practical experience with skip connections, transfer learning, and large datasets. Along the way, we faced challenges such as balancing generator and discriminator training, which deepened our understanding of adversarial learning.

## 📦 Usage 

## 🖌️Color Space Choice

This project uses the LAB color space instead of RGB to separate brightness from color information, simplifying the model’s learning task. For a detailed explanation of why LAB is important and how it benefits colorization, see the project notebook.

## 🗃️ Dataset
The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/shravankumar9892/image-colorization/data)

We use a dataset of 25,000 images in LAB color space, with grayscale (L channel) and color (AB channels) separated. Due to resource constraints, we train on a subset of 10,000 images.

Before training, images are resized and normalized to fit the model input requirements. We also filter out images with extreme brightness or very low colorfulness to improve training quality.

## 🧹 Data Preparation

The dataset was split into training, validation, and test sets. We normalized all channels to the [-1, 1] range for optimal model performance. To enhance generalization, we applied simple data augmentation—random horizontal flips—to the training data. 
For more details, check-out our [notebook](https://github.com/aandrijana/Image-Colorization-Project/blob/main/final_data_preprocessing.ipynb)

## ⌨️ Project Challenges and Key Concepts
### Main Challenges
Our image colorization project presented a range of technical and practical challenges, many of which emerged through trial, error, and iterative learning. Below, we summarize the key issues we faced, organized in the order they arose during development.

- Dataset Quality Limitations: We began by assembling a dataset of color images for training. While sufficient in size, we noticed that the image diversity and resolution were somewhat lacking. Many images lacked rich textures or varied lighting, which may have constrained the model’s ability to generalize — especially for more complex or rare visual scenes.

- Image Size Constraints: Due to storage and memory limitations, particularly on Google Colab, we had to make careful decisions about input resolution. After experimenting with several sizes, we settled on 112×112 pixels. Though this reduced image detail, it allowed us to train on full batches without exceeding memory limits or experiencing crashes.

- Color Distortion from Normalization: In early preprocessing, we applied standard normalization techniques to all channels in the CIELAB color space. However, this introduced a critical issue: color distortion, particularly a strong red tint in both generated and ground truth images. This happened because the a and b channels were scaled improperly, biasing color predictions. Only after correcting the normalization ranges and ensuring proper de-normalization at output did the model produce natural-looking colors.

- Experimentation with CLAHE: To enhance grayscale inputs, we tested CLAHE (Contrast Limited Adaptive Histogram Equalization), which improves local contrast. Surprisingly, this made results worse: the model struggled with the unnatural contrast levels, producing harsh and incoherent outputs. We removed CLAHE from our pipeline after confirming it degraded both visual quality and consistency.

- Training Instability in Google Colab: Most of our training took place in Google Colab, which provided GPU access but also came with frequent runtime disconnections. These interruptions made it difficult to train through full epochs, especially for larger models. To work around this, we split training into three phases, gradually exposing the model to all a and b channels. This staged training helped maintain learning progress despite platform interruptions.

### The Role of Semantics in Image Colorization
Image colorization is a fundamentally ill-posed problem — many grayscale images can correspond to multiple plausible color versions. For instance, a gray shirt could be red, blue, or green; the grayscale image alone doesn't provide enough information to decide. Early methods (e.g., Levin et al.) attempted to address this by using low-level cues like intensity similarity and spatial smoothness, but these are often insufficient in complex scenes.
Semantics — an understanding of what objects are — becomes critical in resolving this ambiguity. Zhang et al. (2016) pioneered the use of deep convolutional neural networks (CNNs) trained on large-scale image datasets to automatically learn the semantic structure of images. Their model doesn’t just look at pixel patterns; it infers object categories and context (e.g., "this region looks like sky", "this is probably a tree", "this is likely a human face").

This semantic insight allows the model to make context-aware color decisions:

Sky regions are predicted to be blue or gray, not green. Trees are colored in various shades of green or brown, depending on season/context. Human skin tones are predicted based on learned priors across diverse examples.

Without this semantic layer, models often produce implausible or jarring results, especially in complex or ambiguous areas of the image. By embedding semantic knowledge, models like Zhang’s can generate colorizations that are not only photorealistic but consistent with human expectations.

In essence, semantics bridge the gap between visual appearance and meaning, enabling more intelligent and believable colorization.

### Generative Adversarial Networks and Pix2Pix for Image Colorization
Generative Adversarial Networks (GANs) have become a powerful tool in image-to-image translation tasks, including image colorization. Introduced by Goodfellow et al. in 2014, GANs consist of two competing neural networks:

A generator that attempts to create realistic images from input data

A discriminator that tries to distinguish between real images (from the training set) and generated (fake) ones

The generator learns to produce increasingly realistic outputs by trying to "fool" the discriminator, while the discriminator becomes better at spotting fakes. This adversarial training process encourages the generator to create outputs that are not only structurally accurate but also visually convincing.

In the context of colorization, the generator takes a grayscale image as input and attempts to generate a plausible color version. The discriminator then evaluates whether the colorized image looks realistic compared to the true color ground truth.



## ⚙️ Model version and optimizers

Firstly, let's take a look at the model architecture:
- Generator encoder architecture:

<img src="https://github.com/user-attachments/assets/fd155bc2-ffe3-4442-8e3b-b667012a0642" width="300"/>
💡 This is the encoder or downsampling path of a U-Net generator. It takes an input image and repeatedly downsamples it, progressively extracting more abstract, high-level features. Each block halves the spatial dimensions and increases the channel depth, compressing the input into a compact feature representation at the "bottleneck".



- Generator decoder architecture:

<img src="https://github.com/user-attachments/assets/f40ac993-64c7-4433-a2cf-0faf2119eb8a" width="300"/>

 💡This is the decoder or upsampling path of the generator, acting as the counterpart to the encoder. It starts with the compressed feature representation from the bottleneck and progressively upsamples it. By doubling the spatial dimensions and reducing the channel depth at each step, it reconstructs the image, translating the learned features back into a high-resolution output.


- Discriminator backbone architecture:

<img src="https://github.com/user-attachments/assets/4f66ab11-edeb-4733-b9ee-a9b3f97b0599" width="300"/>

💡 This is the core feature extraction network for a PatchGAN discriminator. Unlike a traditional discriminator that outputs a single value for the entire image, this model outputs a 2D feature map (e.g., 14x14). Each value in this map corresponds to a different overlapping "patch" of the input image, classifying it as real or fake. This encourages the generator to focus on creating realistic details across the entire image.

## 👀 Results

## 📏 Metrics

## 🏁 Conclusion and future steps
This project successfully implements a Generative Adversarial Network (GAN) to perform automatic image colorization. The model produces vibrant and perceptually realistic colors.  The primary limitation was hardware memory, which constrained the training resolution and model complexity.

Future Steps:
With access to more computational resources, the key priorities are:
- Train on higher-resolution images (228x228) with a deeper model to generate finer details.
- Integrate attention mechanisms to improve contextual understanding and color accuracy.
- With more robust hardware, running the training process for a longer duration. Many GANs require extended training to allow the generator and discriminator to reach a stable Nash equilibrium, resulting in fewer artifacts and more refined details.
- Integrate Fréchet Inception Distance (FID) calculation into the training loop. Then, plot i over time.
  
## 📜 References
- [Levin](https://www.researchgate.net/publication/2896183_Colorization_using_Optimization)

- [Medium article](https://medium.com/data-science/colorizing-black-white-images-with-u-net-and-conditional-gan-a-tutorial-81b2df111cd8)

- [Zhang](https://richzhang.github.io/colorization/)

- [Zhang ECCV Video](https://youtu.be/4xoTD58Wt-0?feature=shared)

- [Stanford](https://cs231n.stanford.edu/reports/2022/pdfs/109.pdf)

- [Anne Guilbert](https://anne-guilbert.medium.com/black-and-white-image-colorization-with-deep-learning-53855922cda6)

## ✒️ License
MIT License
