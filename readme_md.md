# FORMER-DFER Adaptation for MMEW Dataset

This repository provides a PyTorch implementation of the **FORMER-DFER** (Dynamic Facial Expression Recognition via Transformers) architecture adapted for macro-expression recognition on the **MMEW (Multimodal Micro- and Macro-Expression Spotting and Recognition)** dataset.

The spatial-temporal architecture leverages a ResNet backbone combined with Spatial Transformer (`S_Former`) and Temporal Transformer (`T_Former`) modules to extract dynamic facial features across video frame sequences.

---

## 📸 Architecture Overview

The pipeline consists of three primary processing stages:

1. **Spatial Transformer (`S_Former`)**: Processes individual frame spatial features using a ResNet backbone coupled with spatial multi-head self-attention mechanisms.
2. **Temporal Transformer (`T_Former`)**: Captures inter-frame temporal dynamics across video frame sequences (`num_segments=5`) using temporal positional embeddings and a class (`[CLS]`) token.
3. **Classification Head**: A linear dense layer classifying video clips into 6 emotional categories:
   * `anger` (0)
   * `disgust` (1)
   * `fear` (2)
   * `happiness` (3)
   * `sadness` (4)
   * `surprise` (5)

---

## 📁 Repository Structure

```text
.
├── annotation/             # Generated dataset split text files
│   ├── MMEW_macro_train.txt
│   └── MMEW_macro_test.txt
├── dataloader/             # Custom dataset loader and video spatial transformations
│   ├── dataset_MMEW.py
│   └── video_transform.py
├── models/                 # Spatial and Temporal Transformer module implementations
│   ├── S_Former.py
│   ├── T_Former.py
│   └── ST_Former.py
├── log/                    # Training logs, performance charts, and evaluation figures
├── checkpoint/             # Saved model checkpoints (.pth)
├── generate_annotations.py # Dataset parsing and subject-independent split script
├── main_MMEW.py            # Main training loop
├── evaluate_MMEW.py        # Model evaluation and confusion matrix generation
├── evaluate_metrics.py     # WAR, UAR, and per-emotion accuracy evaluation
├── plot_loss_curve.py      # Standalone loss curve rendering
└── README.md               # Project documentation
```

---

## 🛠️ Requirements & Installation

### Prerequisites

* Python 3.8+
* PyTorch 2.0+
* CUDA-capable GPU

### Dependencies

Install the required Python packages:

```bash
pip install torch torchvision numpy scikit-learn matplotlib einops Pillow
```

---

## 🚀 Execution Pipeline

### 1. Dataset Annotation Generation
Prepare subject-independent train/test annotation splits from your raw MMEW dataset directory:

```bash
python generate_annotations.py
```

### 2. Model Training
Train the Spatial-Temporal Transformer model on MMEW macro-expression clips:

```bash
python main_MMEW.py --epochs 100 --batch-size 16 --lr 0.01
```

Training outputs and model checkpoints are stored automatically under `./checkpoint/` and `./log/`.

---

## 📊 Evaluation & Metrics

### 1. Overall Metrics & Per-Emotion Breakdown
Calculate **WAR (Weighted Average Recall)**, **UAR (Unweighted Average Recall)**, and per-emotion accuracy breakdown while exporting a summary bar chart:

```bash
python evaluate_metrics.py
```
*Output Chart:* `./log/performance_metrics_summary.png`

### 2. Confusion Matrix Generation
Evaluate predictions across all 6 expression classes and render the confusion matrix:

```bash
python evaluate_MMEW.py
```
*Output Figure:* `./log/confusion_matrix.png`

---

## 📈 Performance Summary

| Metric | Score |
| :--- | :---: |
| **WAR (Weighted Average Recall / Overall Accuracy)** | **39.58%** |
| **UAR (Unweighted Average Recall / Macro Recall)** | **39.58%** |

### Per-Emotion Accuracy Breakdown

* **Happiness**: 62.5%
* **Fear**: 50.0%
* **Disgust**: 37.5%
* **Sadness**: 37.5%
* **Anger**: 25.0%
* **Surprise**: 25.0%

---

## 📜 References & Acknowledgments

* **FORMER-DFER**: Spatial-Temporal Transformer framework for Dynamic Facial Expression Recognition.
* **MMEW Dataset**: Multimodal Micro- and Macro-Expression Spotting and Recognition Database.