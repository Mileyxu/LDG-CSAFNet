LDG-CSAFNe: Anomaly Detection for MVTec AD
An anomaly detection algorithm implemented based on PyTorch, using a pre-trained ResNet as the backbone network, combined with the CBAM attention mechanism and the Layer Difference Feature Fusion Module (LDCF), performing anomaly scoring in Mahalanobis distance space.

🎯 Project Overview
This project proposes a multi-scale anomaly detection method based on pre-trained features. The main workflow is as follows:

Extract multi-level features using a pre-trained ResNet on ImageNet

Fuse feature differences across different levels using the LDCF (Layer Difference CF) module

Enhance feature representation with the CBAM attention mechanism

Compute anomaly scores using Mahalanobis distance on randomly selected feature subspaces

Apply Gaussian filter smoothing and morphological post-processing to generate the final anomaly segmentation map

✨ Key Features
Multiple backbone networks: Supports ResNet18, ResNet50, WideResNet50_2

Feature fusion strategy: Implements layer difference feature fusion based on LDCF

Efficient inference: Uses random dimensionality sampling to reduce computational complexity

Comprehensive evaluation metrics: Image-level ROCAUC, Pixel-level ROCAUC, PRO AUC, F1-score

Visualization output: Generates anomaly heatmaps, segmentation results, and ROC curves

💻 Environment Requirements
Python >= 3.7

CUDA 10.2+ (optional, GPU recommended)

8GB+ RAM

4GB+ GPU memory (recommended)

🔧 Installation Steps
1. Clone the Repository
```bash
git clone https://github.com/yourusername/LDG-CSAFNe.git
cd LDG-CSAFNe