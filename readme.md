# Alzheimer's Disease Analysis using Federated Learning (Prototype)

## Overview
This repository presents a **baseline implementation** for Alzheimer's disease detection using an Artificial Neural Network (ANN).  
The current work serves as a **centralized, non-pipelined model**, designed as a foundation for future **Federated Learning (FL)** experiments involving distributed and privacy-sensitive clinical data.

## Motivation
Medical data is highly sensitive and often distributed across institutions, making centralized machine learning approaches impractical.  
Federated Learning offers a privacy-preserving alternative by enabling collaborative model training without sharing raw patient data.

This project aims to explore the feasibility of applying Federated Learning to Alzheimer's disease analysis.

## Current Implementation
- Centralized ANN
- Binary classification (Diagnosis)
- Feature preprocessing with encoding and normalization
- Performance evaluation using standard classification metrics

This implementation acts as a **baseline model** for future federated extensions.

## Model Architecture
- Input Layer: Preprocessed clinical features
- Hidden Layers:
  - Dense (4 units, ReLU)
  - Dense (4 units, ReLU)
- Output Layer:
  - Dense (1 unit, Sigmoid)

## Results (Preliminary)
The following results are obtained on a held-out test set and should be treated as **initial experimental observations**.

### Confusion Matrix
![Confusion Matrix](results/confusion_matrix.png)

### Training and Validation Loss
![Loss Curve](results/loss_curve.png)

### Training and Validation Accuracy
![Accuracy Curve](results/accuracy_curve.png)

## Tools & Technologies
- Python
- TensorFlow / Keras
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn

## Planned Extensions (Under Research)
The following aspects are intentionally left open for further research and academic guidance:

- Conversion of the centralized model into a Federated Learning setup
- Simulation of multiple clients representing distributed hospitals
- Analysis of communication efficiency in FL
- Evaluation of client data heterogeneity
- Comparison between pipelined and non-pipelined ANN in federated settings
- Exploration of privacy-preserving mechanisms

## Intended Use
This repository is part of an ongoing research effort and is intended for **academic and research purposes only**.

## Author
**Pratyay Ghose**  
Final Year B.Tech (CSE/IT)  
Techno International Newtown
