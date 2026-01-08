# SEHA
# Learning Cross-modal Self-distillation Hashing with Noisy Labels (SEHA)

Official implementation of the paper **"Learning Cross-modal Self-distillation Hashing with Noisy Labels"**.

## Abstract

Most cross-modal hashing methods assume reliable annotations, which is not always true in real scenarios due to annotation noise. Inspired by human cognitive learning, we propose **SEHA**, a cross-modal self-distillation hashing framework with two core components: **Dual Contrastive Learning (DCL)** and **Self-distillation Learning (SDL)**. SEHA effectively mitigates noisy label effects and achieves state-of-the-art performance.

## Quick Start

### 1. Environment

- Python ≥ 3.8  
- PyTorch ≥ 1.10.0  
- CUDA ≥ 11.3  

Install dependencies:

```bash
pip install numpy pandas scikit-learn tqdm matplotlib

## 2. Quickly Run Experiments

### Single Task

**Example:**

```bash
python train.py --dataset wiki --noisy_ratio 0.8 --bit 164

### Single Task

**Example:**

```bash
python train.py --dataset wiki --noisy_ratio 0.8 --bit 164

### Batch Task

**Example:**

Run preconfigured batch experiments for each dataset:

wiki:
```bash
./run_wiki.bat

INRIA-Websearch:
```bash
./run_INRIA-Websearch.bat

XMedia:
```bash
./run_xmedia.bat

XMediaNet:
```bash
./run_xmedianet.bat
