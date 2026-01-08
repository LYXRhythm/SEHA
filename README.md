# SEHA
# Learning Cross-modal Self-distillation Hashing with Noisy Labels

Official implementation of the paper **"Learning Cross-modal Self-distillation Hashing with Noisy Labels"**.

## Abstract

Most cross-modal hashing (CMH) methods implicitly assume reliable semantic annotations, an assumption that is frequently violated in real-world scenarios due to inevitable annotation noise. Under such noisy supervision, deep neural networks gradually deviate from the reliable knowledge acquired during the early stages of training and increasingly overfit noisy labels as training proceeds, i.e., memorization effect. Existing solutions primarily address noisy supervision from a robust optimization perspective. However, they seldom revisit how the valuable informative representations formed prior to the dominance of noisy memorization can be explicitly retained and exploited. In contrast, human cognitive learning does not passively assimilate incoming information; instead, previously consolidated knowledge is continuously employed as an internal criterion to scrutinize the reliability of extern supervision. For instance, self-directed learners are capable of actively detecting inconsistencies or errors in instructional materials and revising their understanding accordingly. Inspired by this cognitive mechanism, we propose a Cross-modal Self-distillation Hashing framework, termed SEHA. Analogous to human learning behavior, SEHA regards the semantically consistent representations progressively established during the early stages of training as cognitive anchors for subsequent optimization. Specifically, SEHA comprises two tightly coupled components. First, a Dual Contrastive Learning module (DCL) enforces cross-modal consistency at both feature and hash levels, progressively constructing a semantically aligned representation space that serves as an internal reference independent of potentially corrupted annotations. Building upon this foundation, a Self-distillation Learning module (SDL) contrasts external supervision with the model’s historically stabilized predictions, thereby adaptively identifying and attenuating unreliable supervisory signals. Through this iterative self-refinement process, static noisy annotations are effectively transformed into a dynamically evolving supervisory signal. Extensive experiments conducted on four benchmark datasets against 13 representative CMH baselines demonstrate that SEHA mitigates the detrimental effects of noisy labels and achieves state-of-the-art performance.

## Quick Start

### 1. Environment

- Python ≥ 3.8  
- PyTorch ≥ 1.10.0  
- CUDA ≥ 11.3  

Install dependencies:

```bash
pip install numpy pandas scikit-learn tqdm matplotlib
```

## 2. Run Experiments

### Single Task

```bash
python train.py --dataset wiki --noisy_ratio 0.8 --bit 164
```

### Batch Task

Run preconfigured batch experiments for each dataset:

wiki:
```bash
./run_wiki.bat
```

INRIA-Websearch:
```bash
./run_INRIA-Websearch.bat
```

XMedia:
```bash
./run_xmedia.bat
```

XMediaNet:
```bash
./run_xmedianet.bat
```
