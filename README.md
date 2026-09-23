# Beyond I.I.D. Learning: Inductive Biases, Domain Transfer, and Open-Set Recognition

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation, experimental benchmarks, and analysis code for **ATML Assignment 1**. The study investigates the limits of standard Empirical Risk Minimization (ERM) under distribution shifts across four foundational paradigms:
1. **Task 1: Inductive Biases & Feature Representations** (STL-10) — ResNet-50, ViT-B/16, and CLIP evaluated on color distortions, shape-vs-texture cue conflicts, translation stability, and spatial patch shuffling.
2. **Task 2: Unsupervised Domain Adaptation (UDA)** (PACS $\to$ Sketch) — Comparing marginal alignment (DAN, DANN) against class-conditional alignment (CDAN+E), diagnosing negative transfer and domain separability.
3. **Task 3: Domain Generalization (DG)** (PACS $\to$ Unseen Sketch) — Multi-source invariance (DAN-DG) vs. loss-surface parameter flatness (SAM) under strict target quarantine.
4. **Task 4: Open-Set Recognition (OSR)** (CIFAR-10 vs. CIFAR-100) — Post-hoc novelty metrics (MSP, MLS, Energy, Mahalanobis), closed-set representation scaling (GCSC), and manifold-mixup placeholder learning (PROSER).

---

## Repository Structure

```text
ATML-PA-1/
├── common/                     # Shared seeds, logging, and PACS utilities
│   ├── pacs.py
│   └── seed.py
├── task1/                      # Task 1: Inductive Biases (STL-10)
│   ├── data/                   # Subset splitting and cue-conflict generation
│   ├── models/                 # Linear classification heads
│   └── scripts/                # Evaluation scripts (color, cue-conflict, translation, etc.)
├── task2/                      # Task 2: Unsupervised Domain Adaptation (PACS)
│   ├── data/                   # PACS dataloaders
│   ├── methods/                # MMD, GRL, and CDAN multilinear modules
│   ├── models/                 # PACS ResNet-18 backbone & discriminators
│   └── scripts/                # Training, evaluation, and alpha-study scripts
├── task3/                      # Task 3: Domain Generalization (PACS)
│   ├── methods/                # Pairwise source MMD and SAM optimizer
│   └── scripts/                # DG training, evaluation, and rho-study scripts
├── task4/                      # Task 4: Open-Set Recognition (CIFAR-10 / CIFAR-100)
│   ├── data/                   # OSR stratified and near/far splits
│   ├── models/                 # CIFAR-adapted ResNet-18 & PROSER architecture
│   └── scripts/                # Vanilla, GCSC, PROSER training & evaluation
├── report/                     # LaTeX report sources, figures, and plots
├── requirements.txt            # Project dependencies
└── README.md
