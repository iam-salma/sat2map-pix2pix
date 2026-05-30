# 🌍 Sat2Map: Satellite-to-Map Translation with Pix2Pix GAN

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://www.tensorflow.org/)
[![Keras 3](https://img.shields.io/badge/Keras-3.0-red.svg)](https://keras.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Transforming Earth Observation into Navigation-Ready Maps.**  
> This project implements a Conditional Generative Adversarial Network (cGAN) to autonomously translate complex satellite imagery into clean, schematic map representations.

---

## 📖 Table of Contents
- [Project Vision](#-project-vision)
- [Architecture & Theory](#-architecture--theory)
- [Training Evolution](#-training-evolution)
- [Performance Metrics](#-performance-metrics)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Project Vision

Image-to-image translation is a cornerstone of modern computer vision. **Sat2Map** solves the challenge of interpreting raw satellite data by learning the underlying "grammar" of cartography. By training on paired datasets, the model learns to identify roads, buildings, and natural features, rendering them in a standardized map format.

---

## 🏗️ Architecture & Theory

The system is built on the **Pix2Pix** framework, utilizing a two-player game between a **Generator** and a **Discriminator**.

### 1. The Generator (U-Net)
Unlike standard encoders, our Generator uses a **U-Net** architecture. It passes low-level information directly across the network via "skip connections," ensuring that the fine-grained geometry of roads and coastlines is never lost during compression.

### 2. The Discriminator (PatchGAN)
Instead of a simple "real/fake" check for the whole image, we use a **PatchGAN**. It penalizes structure at the scale of local patches ($70 \times 70$), forcing the Generator to create sharp, high-frequency details rather than just blurry approximations.

### 3. Objective Function
The model optimizes a weighted objective:
$$ \arg \min_G \max_D \mathcal{L}_{cGAN}(G, D) + \lambda \mathcal{L}_{L1}(G) $$
Where **$L1$ loss** ensures the generated map is ground-truth aligned, while the **cGAN loss** ensures it looks like a real map.

---

## 📈 Training Evolution

The training process is documented through periodic snapshots (found in `/plots`).

| Phase | Steps | Observation |
| :--- | :--- | :--- |
| **Early** | 1,370 | The model identifies basic color blocks (green for land, blue for water). |
| **Mid** | 6,850 | Road networks begin to appear. The "hallucination" of map symbols becomes consistent. |
| **Late** | 13,700 | Sharp boundaries achieved. Textural noise is replaced by clean vector-like schematic lines. |

*Visual snapshots are saved as `plot_xxxxxx.png` demonstrating the generator "learning" to draw.*

## 🖼️ Visual Results: Final Transformation

The following snapshot showcases the final transformation capability of the Pix2Pix GAN after 13,700 training steps.

<div align="center">
  <img src="plots\labeled\final_result_focused.png" width="500" alt="Final Transformation Results">
  <p><i>Top-to-bottom: Satellite View and Model Generated map.</i></p>
</div>

---

## 📊 Performance Metrics

| Metric | Score | Significance |
| :--- | :--- | :--- |
| **Pixel Accuracy** | **97.05%** | High fidelity in feature placement and color matching. |
| **PSNR** | **27.54 dB** | Signal-to-noise ratio indicating low reconstruction error. |
| **SSIM** | **0.7255** | Strong structural similarity to ground truth maps. |
| **MAE** | **0.0273** | Minimal average pixel-wise deviation. |

---

## 🛠️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/your-username/sat2map-pix2pix.git
cd sat2map-pix2pix

# Install dependencies
pip install -r requirements.txt
```

---

## ⚡ Usage

### Training & Analysis
Open `train.ipynb` to view the full research lifecycle, from data loading to advanced evaluation heatmaps.

### Production Inference
```python
from keras.models import load_model
import numpy as np

# Load trained weights
model = load_model('models/model_013700.h5')

# Translate
satellite_patch = load_my_image() # (256, 256, 3)
generated_map = model.predict(satellite_patch)
```

---

## 🤝 Acknowledgments

- **Dataset:** UC Berkeley [Maps Dataset](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz).
- **Inspiration:** The seminal work by Isola et al. on Conditional Adversarial Networks.
- **Support:** Core architectural patterns adapted from Jason Brownlee's GAN research.

