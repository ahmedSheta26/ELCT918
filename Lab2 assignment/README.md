# ELCT918: Selected Topics in AI Accelerators Hardware Design
## Lab Assignment 2: Comparative Study of Classical CNN Architectures (LeNet-5, AlexNet, VGG16)

**Student Name:** Ahmed Mohamed Ahmed Ali Sheta  
**Student ID:** 58-24418  
**Course:** ELCT918 — Selected Topics in AI Accelerators Hardware Design  
**Instructors:** Dr. Eman Azab, MSc Eng. Nour ElShahawy  
**Repository Path:** `ELCT918/Lab2 assignment/`

---

## 1. Project Overview

This project implements, evaluates, and compares three foundational Convolutional Neural Network (CNN) architectures—**LeNet-5** (LeCun et al., 1998), **AlexNet** (Krizhevsky et al., 2012), and **VGG16** (Simonyan & Zisserman, 2014)—trained **from scratch** on:
1. **CIFAR-10** (10 classes, $32\times32$ RGB images)
2. **CIFAR-100** (100 classes, $32\times32$ RGB images)

The objective is to analyze how architectural choices (filter sizes, network depth, channel width, parameter volume, and receptive field expansion) impact generalization, computational overhead, and empirical susceptibility to overfitting and underfitting across distinct task complexities.

---

## 2. Framework & Dependencies

- **Framework:** PyTorch (`torch`, `torchvision`)
- **Language:** Python 3.8+
- **Hardware Acceleration:** CUDA-compatible GPU (recommended) or CPU

### Prerequisites & Installation

Create a virtual environment and install the required dependencies:

```bash
# Optional: create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision matplotlib
```

---

## 3. Repository Structure

```text
Lab2 assignment/
├── models.py                     # Implementations of LeNet-5, AlexNet, and VGG16
├── train.py                      # Training, validation, testing pipeline, and curve plotting
├── results/                      # Generated evaluation outputs and figures
│   ├── summary.csv               # Consolidated metrics for all 6 model-dataset combinations
│   ├── lenet_cifar10_curves.png  # Loss & accuracy curves: LeNet-5 on CIFAR-10
│   ├── lenet_cifar10_history.json
│   ├── lenet_cifar100_curves.png # Loss & accuracy curves: LeNet-5 on CIFAR-100
│   ├── lenet_cifar100_history.json
│   ├── alexnet_cifar10_curves.png
│   ├── alexnet_cifar10_history.json
│   ├── alexnet_cifar100_curves.png
│   ├── alexnet_cifar100_history.json
│   ├── vgg16_cifar10_curves.png
│   ├── vgg16_cifar10_history.json
│   ├── vgg16_cifar100_curves.png
│   └── vgg16_cifar100_history.json
├── Lab2_Report.pdf               # Comprehensive analytical report
└── README.md                     # Documentation and execution guide
```

---

## 4. Architectural Implementations & CIFAR Adaptations

All models were implemented from scratch without pre-trained weights in `models.py`. Due to the low-resolution nature of CIFAR ($32\times32$) compared to original paper inputs ($224\times224$ for AlexNet/VGG16), specific structural adaptations were applied:

### Summary of Architectural Adaptations

| Architecture | Component | Original Paper | Lab Implementation | Justification / Adaptation Reason |
| :--- | :--- | :--- | :--- | :--- |
| **LeNet-5** | Input Channels | 1 (Greyscale $28\times28$ / $32\times32$) | 3 (RGB $32\times32$) | Accepts 3-channel CIFAR color inputs directly. |
| **LeNet-5** | C3 Layer Connections | Sparse connection matrix | Fully connected to all 6 S2 maps | Simplifies graph definition; modern compute easily handles the resulting 2,416 weights. |
| **LeNet-5** | Subsampling (S2, S4) | Trainable coefficients + sigmoid | `nn.AvgPool2d(2, stride=2)` | Matches standard PyTorch pooling primitives. |
| **LeNet-5** | Classifier Output | Radial Basis Functions (RBF) | `nn.Linear(84, num_classes)` | Standard linear classification head compatible with Cross-Entropy Loss. |
| **AlexNet** | Conv1 Filter & Stride | $11\times11$, stride 4 | $5\times5$, stride 1, pad 2 | An $11\times11$ filter with stride 4 on $32\times32$ aggressively shrinks spatial maps to $6\times6$ immediately, losing critical fine spatial detail. |
| **AlexNet** | FC6 Input Size | $256 \times 6 \times 6 = 9,216$ | $256 \times 3 \times 3 = 2,304$ | Direct downstream effect of the $32\times32$ input geometry. |
| **AlexNet** | GPU Partitioning | Split across 2 GPUs | Single consolidated network | The original 2-GPU split addressed 2012 VRAM limits (3 GB); unnecessary on modern accelerators. |
| **VGG16** | Feature Reduction | 5 pooling layers on $224\times224 \to 7\times7$ | 5 pooling layers on $32\times32 \to 1\times1$ | Standard Configuration D kept intact; output entering FC6 becomes $512\times1\times1 = 512$ (vs 25,088). |
| **AlexNet & VGG16**| Weight Initialization | Gaussian / Pre-trained shallow nets | Kaiming Normal (`kaiming_normal_`) | Ensures stable variance across deep ReLU layers without multi-stage manual pre-training. |

### Layer and Parameter Summary

| Model | Weight Layers (Conv + FC) | Total Layers | Parameters (CIFAR-10) | Parameters (CIFAR-100) |
| :--- | :---: | :---: | :---: | :---: |
| **LeNet-5** | 5 (3 conv + 2 FC) | 7 | 62,006 | 69,656 |
| **AlexNet** | 8 (5 conv + 3 FC) | 11 | 29,983,114 | 30,351,844 |
| **VGG16** | 16 (13 conv + 3 FC) | 21 | 33,638,218 | 34,006,948 |

---

## 5. Training Setup & Hyperparameters

To ensure an unbiased architecture-driven comparison, all 6 runs use identical hyperparameters:

- **Dataset Partitioning:** 45,000 training, 5,000 validation (fixed deterministic split via seed 42), 10,000 test.
- **Normalization:** Per-channel mean and standard deviation scaling.
- **Optimizer:** Stochastic Gradient Descent (SGD)
  - Learning Rate ($\eta$): `0.01`
  - Momentum: `0.9`
  - Weight Decay: `5e-4`
- **Batch Size:** 128
- **Epochs:** 30
- **Data Augmentation:** Disabled (`--augment` flag kept off by default to isolate pure architectural capacity and facilitate observable overfitting dynamics).
- **Model Checkpointing:** Best validation accuracy weights are retained for the final test set evaluation.

---

## 6. How to Run

### Sanity Check (Inspect Dimensions & Parameter Counts)
Run `models.py` directly to verify output shapes and parameter tables:
```bash
python models.py
```

### Quick Verification Run
Test a single model on CIFAR-10 for 2 epochs:
```bash
python train.py --model lenet --dataset cifar10 --epochs 2
```

### Full Benchmark (All 6 Runs)
Execute training across all three architectures and both datasets for 30 epochs:
```bash
python train.py --model all --dataset all --epochs 30
```

### Custom Model & Dataset Flags
Train a specific architecture on a targeted dataset:
```bash
# Train AlexNet on CIFAR-100
python train.py --model alexnet --dataset cifar100 --epochs 30

# Train VGG16 with Batch Normalization enabled (optional stability check)
python train.py --model vgg16 --dataset cifar100 --epochs 30 --vgg-bn
```

All summary metrics are written to `results/summary.csv`, and curves are saved to `results/<model>_<dataset>_curves.png`.

---

## 7. Experimental Results & Performance Summary

Accuracy Drop is defined as:
$$\text{Accuracy Drop (\%)} = 100\% - \text{Test Top-1 Accuracy (\%)}$$

### Summary Benchmark Table (from `results/summary.csv`)

| Architecture | Dataset | Test Top-1 (%) | Test Top-5 (%) | Accuracy Drop (%) | Final Train Acc (%) | Final Val Acc (%) | Train-Val Gap (%) | Avg Epoch Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LeNet-5** | CIFAR-10 | 60.51 | 95.73 | 39.49 | 71.82 | 59.50 | 12.32 | 10.78 |
| **LeNet-5** | CIFAR-100 | 27.57 | 56.78 | 72.43 | 34.02 | 27.84 | 6.18 | 11.28 |
| **AlexNet** | CIFAR-10 | 78.82 | 98.45 | 21.18 | 98.72 | 78.62 | 20.10 | 27.34 |
| **AlexNet** | CIFAR-100 | 50.27 | 77.26 | 49.73 | 93.21 | 50.38 | 42.83 | 28.13 |
| **VGG16** | CIFAR-10 | 80.47 | 98.38 | 19.53 | 99.31 | 79.44 | 19.87 | 21.24 |
| **VGG16** | CIFAR-100 | 43.19 | 70.91 | 56.81 | 92.22 | 42.62 | 49.60 | 21.78 |

---

## 8. Key Analytical Findings

1. **Impact of Task Complexity (CIFAR-10 vs CIFAR-100):**
   - Every architecture shows a sharp degradation when transitioning from 10 to 100 classes. Top-1 test accuracy drops by **32.94%** for LeNet-5, **28.55%** for AlexNet, and **37.28%** for VGG16.
   - For CIFAR-100, the available training samples per class drop from ~4,500 down to ~450, simultaneously increasing inter-class visual similarity.

2. **Underfitting vs Overfitting:**
   - **LeNet-5 suffers from underfitting:** With only 6 to 16 convolutional feature maps, its representational capacity is insufficient for 100 classes, yielding only 34.02% training accuracy and an accuracy drop of 72.43%.
   - **AlexNet and VGG16 suffer from severe overfitting:** Both architectures easily fit the training set (>92% train accuracy), but display large generalization gaps. VGG16 exhibits the largest train-val gap on CIFAR-100 (**49.60 percentage points**), demonstrating that excess capacity without explicit regularization or data augmentation impairs performance on data-constrained regimes.

3. **Diminishing Returns with Extreme Depth:**
   - On CIFAR-10, doubling weight layers from AlexNet (8 layers) to VGG16 (16 layers) yields an accuracy increase of only **+1.65 percentage points** (78.82% $\to$ 80.47%).
   - On CIFAR-100, VGG16 is outperformed by AlexNet by **7.08 percentage points** (43.19% vs 50.27%), confirming that deeper, parameter-dense architectures require proportional data scaling or modern stabilization techniques (e.g., BatchNorm, aggressive augmentation) to realize their theoretical depth advantage.

---

## 9. References

- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-based learning applied to document recognition*. Proceedings of the IEEE, 86(11), 2278-2324.
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). *ImageNet classification with deep convolutional neural networks*. Advances in Neural Information Processing Systems (NeurIPS 2012).
- Simonyan, K., & Zisserman, A. (2014). *Very deep convolutional networks for large-scale image recognition*. International Conference on Learning Representations (ICLR 2015).