# FIS-BR: Feature-Importance-Aware SMOTE and Balanced Replay for Continual Depression Severity Recognition

## Introduction

FIS-BR is a continual learning framework for video-based depression severity recognition under highly imbalanced and continuously evolving clinical data streams.

The framework is built upon the STAD-GWR backbone and integrates:

- **STAD**: Spatial-Temporal Attention Distillation
- **GWR**: Gradient-Weight-Aware Regularization
- **BR**: Attention-Guided Balanced Replay
- **FIS**: Feature-Importance-Aware SMOTE

to simultaneously alleviate catastrophic forgetting and severe class imbalance in continual depression diagnosis.

The framework is specifically designed for long-term clinical deployment scenarios where new patient data continuously arrive and severe depression samples are extremely scarce.

---

# Framework Overview

The proposed framework contains four major components:

## 1. Spatial-Temporal Attention Distillation (STAD)

STAD transfers spatial and temporal attention knowledge from previous tasks to the current model by minimizing discrepancies between attention maps across tasks, helping preserve critical depression-related spatio-temporal patterns.

## 2. Gradient-Weight-Aware Regularization (GWR)

GWR estimates parameter importance using both gradient sensitivity and parameter magnitude, and constrains updates on important parameters to reduce catastrophic forgetting.

## 3. Balanced Replay (BR)

BR constructs an attention-guided memory buffer using:

- prototype-based replay
- uncertainty-aware replay
- balanced class sampling

to preserve representative historical samples under limited memory budgets.

## 4. Feature-Importance-Aware SMOTE (FIS)

FIS performs constrained interpolation in discriminative feature subspaces to synthesize high-quality severe depression samples, improving minority class representation while preserving semantic consistency.

---

# Features

- Continual learning for depression video analysis
- Attention-guided replay memory
- Severe class augmentation
- Spatio-temporal attention distillation
- Long-tailed continual learning
- Replay-based anti-forgetting strategy
- Compatible with AVEC2013 and AVEC2014

---
# 运行代码

```bash
python3 -u src/main.py --network STA --approach ours --num-tasks 4 --nepochs 500 --log disk --batch-size 5 --gpu 0 --exp-name fis_br_exp --lr 0.001 --seed 1 --lamb 1.0 --lr-patience 20 --plast_mu 1.0 --pool-along spatial --br-enable --br-memory-size 200 --br-proto-ratio 0.8 --br-uncert-ratio 0.2 --br-synthetic-ratio 0.3 --fis-enable --fis-lambda-min 0.2 --fis-lambda-max 0.8 --fis-topk-ratio 0.3
