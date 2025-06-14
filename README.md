# Image-Colorization-Project
Students: Andrijana Kešelj, Anđela Maksimović

This is the final version of the image colorization project done by students of DSAI ETF Sarajevo.

Important note: Due to inability to delete past commits and some errors present in our previous GitHub repository's files, we decided to push the whole project only after completing everything.


# 🎨 ColorizeIMG

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
- [🔮 Conclusion and future steps](#-conclusion-and-future-steps)
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
During this project, we encountered several challenges:

- Limited dataset quality and image resolution constrained model generalization.

- Due to Colab resource limits, we trained on 112×112 images.

- Early normalization caused color distortions (reddish tint) which we fixed by correcting scaling.

- CLAHE contrast enhancement degraded results and was removed.

- Training on Colab was interrupted frequently; we split training into phases to cope.

- Image colorization is an inherently ambiguous problem requiring semantic understanding of scenes. Models like Zhang et al. leverage deep learning to infer object context, improving color accuracy.

We also explored Generative Adversarial Networks (GANs), especially the Pix2Pix conditional GAN framework, which uses adversarial training for more realistic colorization results.

## ⚙️ Model version and optimizers

## 👀 Results

## 📏 Metrics

## 🔮 Conclusion and future steps

## 📜 References
- [Levin](https://www.researchgate.net/publication/2896183_Colorization_using_Optimization)

- [Medium article](https://medium.com/data-science/colorizing-black-white-images-with-u-net-and-conditional-gan-a-tutorial-81b2df111cd8)

- [Zhang](https://richzhang.github.io/colorization/)

- [Zhang ECCV Video](https://youtu.be/4xoTD58Wt-0?feature=shared)

- [Stanford](https://cs231n.stanford.edu/reports/2022/pdfs/109.pdf)

- [Anne Guilbert](https://anne-guilbert.medium.com/black-and-white-image-colorization-with-deep-learning-53855922cda6)

## ✒️ License
MIT License
