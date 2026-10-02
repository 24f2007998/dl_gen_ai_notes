# Deep Learning & Generative AI — Course Notes

Comprehensive, structured study notes and reference material for **Week 1 (Lectures 1–3)**.

---

## Table of Contents
- [Instructors & Overview](#instructors--overview)
- [Lecture 1: Transition from Classical Machine Learning to Deep Learning](#lecture-1-transition-from-classical-machine-learning-to-deep-learning)
  - [1. Prerequisites](#1-prerequisites)
  - [2. Foundations of Machine Learning](#2-foundations-of-machine-learning)
  - [3. Problem Types and Input-Output Modalities](#3-problem-types-and-input-output-modalities)
  - [4. The ML vs. DL Paradigm Shift](#4-the-ml-vs-dl-paradigm-shift)
  - [5. Catalysts of the Deep Learning Revolution](#5-catalysts-of-the-deep-learning-revolution)
- [Lecture 2: PyTorch Fundamentals & Tensor Architecture](#lecture-2-pytorch-fundamentals--tensor-architecture)
  - [1. What is PyTorch?](#1-what-is-pytorch)
  - [2. Tensor Dimensions & Physical Representations](#2-tensor-dimensions--physical-representations)
  - [3. Mathematical Notation Conventions](#3-mathematical-notation-conventions)
- [Lecture 3: PyTorch Environment & CUDA Acceleration](#lecture-3-pytorch-environment--cuda-acceleration)
  - [1. System Requirements & Dependencies](#1-system-requirements--dependencies)
  - [2. Environment Verification & GPU Diagnostics Script](#2-environment-verification--gpu-diagnostics-script)
- [Quick Reference Cheat Sheet](#quick-reference-cheat-sheet)

---

## Instructors & Overview
- **Instructors:** Prof. Ganapathi & Prof. Balaji Srinivasan  
- **Department:** Department of Data Science and Artificial Intelligence  
- **Structure:** 
  - First module: Machine learning foundations, deep neural network fundamentals, and practical PyTorch mechanics.
  - Later modules: Advanced architectures and Generative AI pipelines.

---

## Lecture 1: Transition from Classical Machine Learning to Deep Learning
*Lecture Video:* [YouTube Link](https://youtu.be/Z7t12UH0cyY)

### 1. Prerequisites
Before studying deep learning implementations, verify familiarity with the following areas:

1. **Applied Machine Learning:**
   - Supervised learning workflows
   - Linear regression and regularized models
   - Logistic regression and multi-class softmax classification
2. **Mathematical Foundations:**
   - **Calculus:** Multivariate gradients and the **Chain Rule** (essential for computing backpropagation gradients).
   - **Linear Algebra:** Vector spaces, matrix transformations, dot products, and matrix-vector multiplications.
3. **Core Python Tooling:**
   - Numerical computation: `numpy`
   - Data structures: `pandas`
   - Data visualization: `matplotlib`
   - Deep learning computation: `torch` (PyTorch)
   - Execution environment: **Google Colab** (recommended for zero-config GPU access) or local Jupyter installations.

---

### 2. Foundations of Machine Learning

> **Arthur Samuel (1959):** *"Machine Learning is the field of study that gives computers the ability to learn without being explicitly programmed."*

In rule-based programming, software engineers manually write logic via conditionals and heuristics. In machine learning, systems deduce the underlying patterns directly from structured or unstructured training data.

```
       +-----------------------+
Input  |                       |  Prediction
  X -->|  Model Parameters     |-----> y_hat
       |  (Weights W, Bias b)  |         |
       +-----------------------+         v
                   ^             [ Loss Function ]
                   |              (y_hat vs. y)
                   |                     |
                   +--- Feedback Update -+
```

1. **Input Matrix ($X$):** Data samples fed into the estimator.
2. **Model Parameters ($\mathbf{W}, b$):** Latent variables that capture underlying representations.
3. **Loss Function:** Quantifies the divergence between actual ground truth ($y$) and predictions ($\hat{y}$).
4. **Parameter Update:** Uses calculated errors to iteratively adjust internal weights through optimization routines.

---

### 3. Problem Types and Input-Output Modalities

| Use Case | Input Format ($X$) | Modality Type | Output Target ($y$) | Target Structure |
| :--- | :--- | :--- | :--- | :--- |
| **Real Estate Valuation** | Tabular database records (sqft, bedrooms, postal code) | Structured Data | Numerical sale price | Continuous Scalar ($\mathbb{R}$) |
| **Autonomous Driving** | Real-time multi-angle camera feeds | Computer Vision | Steering angle, brake pressure, speed | Continuous Vector ($\mathbb{R}^k$) |
| **Sentiment / Spam Analysis** | Natural language text sequences | Unstructured NLP | Classification label ($0$ or $1$) | Discrete Categorical |
| **Content Recommendations** | Sparse user interaction history logs | Tabular / Sparse matrix | Item preference ranking | Ranked Probability Distribution |

---

### 4. The ML vs. DL Paradigm Shift

```
Classical Machine Learning:
[ Raw Input ] ---> [ Manual Feature Engineering ] ---> [ Shallow Classifier ] ---> Prediction

Deep Learning (End-to-End):
[ Raw Input ] ------------> [ Deep Neural Network Hierarchy ] ------------> Prediction
                            (Learns features + classifier together)
```

#### Classical Machine Learning
- **Manual Feature Engineering Bottleneck:** Requires domain experts to handcraft mathematical features (ratios, PCA projections, edge filters) before models can train successfully.
- **High-Dimensional Limitations:** Standard estimators struggle on complex arrays like images or audio because manual feature representations cannot capture complex contextual distributions.
- **Performance Saturation:** Model accuracy often hits a plateau, failing to scale even when massive new datasets are supplied.

#### Deep Learning (Hierarchical Representation Learning)
- **Automatic Feature Learning:** Neural network layers extract representations autonomously from raw tensors.
  - *Early layers:* Capture primitive low-level features (edges, color boundaries, sharp gradients).
  - *Intermediate layers:* Group primitives into parts (textures, corners, geometric contours).
  - *Deep layers:* Construct semantic, task-specific abstractions (complete objects, faces, syntactic structures).
- **End-to-End Optimization:** Raw inputs map directly to targeted outcomes without fragmented intermediary heuristics.
- **Continuous Scaling:** Performance consistently scales with increased model capacity and data volume.

---

### 5. Catalysts of the Deep Learning Revolution

The concepts behind artificial neural networks date back decades. Modern deep learning became dominant due to three simultaneous shifts:

```
                      +-------------------+
                      |     Big Data      |
                      | (ImageNet, Web)   |
                      +---------+---------+
                                |
                                v
+-------------------+   +---------------+   +--------------------+
| Hardware Engines  |-->| Deep Learning |<--| Algorithm Upgrades |
| (NVIDIA GPU/TPUs) |   |  Breakthrough |   | (Adam, ReLU, etc.) |
+-------------------+   +---------------+   +--------------------+
```

1. **Big Data:** Large-scale annotated datasets (such as ImageNet, Common Crawl, and Wikipedia dumps) provided the supervisory signal needed to train millions of parameters without immediate overfitting.
2. **Specialized Compute Hardware:** Repurposing Graphics Processing Units (GPUs) for high-throughput, parallel floating-point matrix multiplications reduced training runs from months to hours.
3. **Algorithmic Advancements:**
   - Mitigating vanishing gradients using **ReLU** and modern activation functions.
   - Robust optimization algorithms with adaptive momentum (e.g., Adam, AdamW).
   - High-capacity architectures (Convolutional Networks, ResNets, Transformers).

```
+--------------------------------------------------------+
| Artificial Intelligence (AI)                           |
|   +--------------------------------------------------+ |
|   | Machine Learning (ML)                            | |
|   |   +--------------------------------------------+ | |
|   |   | Deep Learning (DL)                         | | |
|   |   |   +--------------------------------------+ | | |
|   |   |   | Generative AI (LLMs, Diffusion)      | | | |
|   |   |   +--------------------------------------+ | | |
|   |   +--------------------------------------------+ | |
|   +--------------------------------------------------+ |
+--------------------------------------------------------+
```

---

## Lecture 2: PyTorch Fundamentals & Tensor Architecture
*Lecture Video:* [YouTube Link](https://youtu.be/jG_YEftvVdE)

### 1. What is PyTorch?
PyTorch combines two fundamental systems:
- An accelerated **scientific computing library** featuring an interface similar to NumPy, with complete hardware acceleration on CUDA-enabled GPUs.
- A **deep learning framework** with dynamic computational graphs and a tape-based automatic differentiation engine (`torch.autograd`).

### 2. Tensor Dimensions & Physical Representations
A **Tensor** is a generalization of scalars, vectors, and matrices to an arbitrary number of dimensions ($N$-D arrays).

| Order / Rank | Mathematical Concept | Description | Structural Dimension | Practical Deep Learning Use Case |
| :---: | :---: | :---: | :---: | :---: |
| **$0\text{D}$** | Scalar | A single numerical value | `()` | Loss metric values, scalar learning rate |
| **$1\text{D}$** | Vector | A 1D sequence of numbers | `(D,)` | Bias term vector, single sample tabular features |
| **$2\text{D}$** | Matrix | A 2D grid ($H \times W$) | `(R, C)` | Linear layer weight matrix $\mathbf{W}$, grayscale image |
| **$3\text{D}$** | 3D Tensor | A cube / volume array | `(C, H, W)` | Single RGB color image (Channels, Height, Width) |
| **$4\text{D}$** | 4D Tensor | A batch of volumes | `(B, C, H, W)` | Standard mini-batch of RGB images; or single video clip `(T, C, H, W)` |
| **$5\text{D}$** | 5D Tensor | A multi-batch sequence | `(B, T, C, H, W)` | Mini-batch of video sequences (Batch, Time/Frames, Channels, Height, Width) |

### 3. Mathematical Notation Conventions
- **Scalars:** Lowercase italic letters ($x, y, w, b$).
- **Vectors:** Lowercase bold letters ($\mathbf{x}, \mathbf{y}, \mathbf{w}$).
- **Matrices:** Uppercase bold letters ($\mathbf{X}, \mathbf{W}$).
- **Tensors:** Multidimensional arrays defined over real vector spaces:
  $$\mathbf{X} \in \mathbb{R}^{d_1 \times d_2 \times \dots \times d_n}$$
  Where $d_i$ defines the cardinality along dimension axis $i$.

---

## Lecture 3: PyTorch Environment & CUDA Acceleration
*Lecture Video:* [YouTube Link](https://youtu.be/L3YERS0wRew)

### 1. System Requirements & Dependencies
- **Interpreter:** Python 3.7 or newer.
- **Host Memory:** Minimum 4 GB RAM recommended for basic workflows.
- **Hardware Acceleration:** NVIDIA GPU supporting CUDA drivers (optional for CPU-only experiments, recommended for training deep models).
- **Interoperability:** Seamless memory conversion between NumPy arrays and PyTorch tensors via `torch.from_numpy()` and tensor `.numpy()` methods without memory copying when possible.

---

### 2. Environment Verification & GPU Diagnostics Script

Use this utility script to verify runtime dependencies, inspect GPU availability, and read device properties:

```python
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def run_environment_diagnostics():
    print("=" * 55)
    print("       ENVIRONMENT DIAGNOSTICS & SYSTEM STATUS       ")
    print("=" * 55)
    
    # 1. Dependency Versions
    print(f"PyTorch Version    : {torch.__version__}")
    print(f"NumPy Version      : {np.__version__}")
    print(f"Matplotlib Version : {matplotlib.__version__}")
    print("-" * 55)
    
    # 2. CUDA Acceleration Verification
    cuda_ready = torch.cuda.is_available()
    print(f"CUDA Hardware Acceleration Available: {cuda_ready}")
    
    if cuda_ready:
        device_id = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_id)
        props = torch.cuda.get_device_properties(device_id)
        total_vram_gb = props.total_memory / (1024 ** 3)
        
        print(f"Active Device ID         : {device_id}")
        print(f"GPU Hardware Model       : {device_name}")
        print(f"Compute Capability       : {props.major}.{props.minor}")
        print(f"Total Dedicated VRAM     : {total_vram_gb:.2f} GB")
    else:
        print("Running on CPU mode. No CUDA-compatible GPU detected.")
    print("=" * 55)

if __name__ == "__main__":
    run_environment_diagnostics()
```

---

## Quick Reference Cheat Sheet

```
+------------------------------------------------------------------------------------+
|                                CORE CHEAT SHEET                                    |
+------------------------------------------------------------------------------------+
| ML Bottleneck        : Manual feature extraction and scaling limits               |
| DL Breakthrough      : End-to-end automated hierarchical representation learning   |
| Three Drivers        : Big Data + GPU Acceleration + Modern Optimization Methods   |
+------------------------------------------------------------------------------------+
| Scalar (0D)          : torch.tensor(3.14)                                         |
| Vector (1D)          : torch.tensor([1.0, 2.0, 3.0])                              |
| Matrix (2D)          : torch.tensor([[1, 2], [3, 4]])                             |
| RGB Image (3D)       : Shape: (3, Height, Width)                                  |
| Batched Images (4D)  : Shape: (Batch_Size, 3, Height, Width)                      |
| Batched Video (5D)   : Shape: (Batch_Size, Frames, 3, Height, Width)               |
+------------------------------------------------------------------------------------+
| CUDA Inspection      : torch.cuda.is_available()                                  |
| Target Device        : torch.device("cuda" if torch.cuda.is_available() else "cpu")|
+------------------------------------------------------------------------------------+
```