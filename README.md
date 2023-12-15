# Time Series Segmentation

## Overview

The "Time Series Segmentation" project provides a comprehensive solution for computing breakpoints on multivariate time series using a pipeline enriched with machine learning features. This tool allows users to create a model, train it for specific segmentation tasks, and leverage machine learning techniques for improved accuracy.

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
- [Usage](#usage-instructions)
- [Experimentation](#Experimentation)
  - [CNN Architecture](#cnn-architecture)
  - [FCNN Architecture](#fcnn-architecture)

## Getting Started

### Prerequisites

To use the databases employed in this project, you are required to download the "Databases" folder. Specify the path to access this folder in the `main_constant.py` file on your machine.

[Databases Folder](https://drive.google.com/drive/folders/17pU6Sy50mGwbkWo1lSW54WkzouJDnRTy?usp=sharing)

### Usage Instructions

All the necessary information for using this repository is explained with examples in the notebook `example.ipynb`. Refer to this notebook for a detailed guide on how to use the functionalities provided by this repository.

You can also read this PDF summarizing the project and its mathematical aspects: [Project Summary PDF](https://drive.google.com/file/d/1BVJmkIo9FJL8CaKSf6toWkZ0KeVRvUIC/view?usp=sharing)

## Experimentation

This repository is associated with a scientific paper ([LINK](#)) where the results are presented. To obtain these results, we utilized the benchmark described in the `benchmark.py` file. Feel free to execute these benchmarks yourself (To do so, specify the destination of your results in line 184).

All pipelines used in this benchmark are constructed in the `pipelines.py` file. To understand how it is structured, you can follow the example notebook provided in the repository.

The following is a quick explanation of the two main architectures: CNN and FCNN.

### CNN Architecture

The Convolutional Neural Network (CNN) employed in our approach consists of two consecutive convolutional layers designed for effective feature extraction, with each convolution followed by a ReLU activation.

The convolutional layers are characterized by the following parameters:

- **Stride (2):** In the context of CNNs, the stride parameter determines the step size of the convolutional kernel as it traverses the input data during convolution operations. With a stride of 2, the kernel moves four units at a time, effectively down-sampling the feature maps. The primary reason for using this stride is to reduce the size of the transformed signal, enabling rapid segmentation while retaining essential information for accuracy.

- **Temporal Length (300):** The temporal length represents the duration of sequences or time steps processed by the network. Each input sequence consists of 300 time steps.

- **Input Dimension (3):** This parameter defines the spatial dimensions of the convolutional kernel. Here, it indicates a 3x3 kernel size for our 1D convolutional operations.

- **Output Dimension (3 and 2):** The output dimension specifies the number of filters or feature maps generated by each convolutional layer.

    - In the first convolutional layer, the output dimension is set to 3.
    
    - In the second convolutional layer, the output dimension is reduced to 2, aligning with binary classification requirements.

After the two convolutional layers, we ensure the signal falls within the range of 0 to 1 for uniformity and improved performance.

![CNN Architecture](https://drive.google.com/uc?id=1WPqmqH9_5kS6zyGgwnobOwrMM8wMH2mh)

### FCNN Architecture

The FCNN architecture includes two consecutive convolutional layers followed by two linear layers for feature extraction, each convolution followed by a ReLU activation. The convolutional layers are characterized by:

- **First Convolution (300, n_dims, n_dims):** Represents a convolutional layer with a temporal length of 300, input dimension `n_dims`, and `n_dims` output dimensions.
  
- **Second Convolution (300, n_dims, n_dims):** Similar to the first layer.

The linear layers are characterized by:

- **First Linear Layer (n_dims, 10):** Specifies linear layer sizes, including a dense layer with `n_dims` input dimensions and 10 output dimensions.

- **Second Linear Layer (10, 2):** Specifies linear layer sizes, including a dense layer with 10 input dimensions and 2 output dimensions.

Like the CNN network, we ensure the signal falls within the 0 to 1 range before performing the classic segmentation method.

![FCNN Architecture](https://drive.google.com/uc?id=1UyfSP9D64JUamIEIUqXt9KkhhPAMtFox)

