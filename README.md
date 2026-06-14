# Comparative Evaluation of Tree-Based and Deep Learning Hybrid Architectures for Explainable Zero-Day Intrusion Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

This repository contains the official implementation and experimental code for the research paper: *"Comparative Evaluation of Tree-Based and Deep Learning Hybrid Architectures for Explainable Zero-Day Intrusion Detection"*.

## 📖 Abstract
Network intrusion detection systems face significant challenges in identifying zero-day attacks. This repository provides the code for a unified, three-stage hybrid framework: (1) binary attack detection, (2) multi-class classification of known attacks, and (3) unsupervised autoencoder-based anomaly detection for identifying zero-day attacks. We rigorously benchmark five diverse architectures (XGBoost, Random Forest, Decision Tree, CNN, and LSTM) combined with eight distinct feature selection methods on the UNSW-NB15 dataset, under a strict zero-day simulation protocol where three attack families are completely withheld from training.

## 📂 Repository Structure
```text
├── 01_pre-processing.py          # Data cleaning, encoding, scaling, and train/test splitting
├── 02_model_training           # the 5 hybrid model scripts
│   ├── xgboost_hybrid_1.py
│   ├── random_forest_hybrid_1.py
│   ├── decision_tree_hybrid_1.py
│   ├── cnn_hybrid_1.py
│   └── lstm_hybrid_1.py
├── 03_post_process.py        # Aggregates CSV outputs and generates heatmaps/ROC curves
├── requirements.txt             # Python dependencies
├── README.md                    # This file
