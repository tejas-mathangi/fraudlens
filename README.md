cat > README.md << 'EOF'
# FraudLens

A Graph Neural Network based fraud detection system built on the Elliptic Bitcoin Dataset.

## Overview
FraudLens detects coordinated financial fraud using GraphSAGE, community detection, and GNNExplainer to identify fraudulent accounts, fraud rings, and explain suspicious patterns.

## Stack
- Python 3.12
- PyTorch 2.2.2 + PyTorch Geometric 2.7.0
- Streamlit (dashboard)
- NetworkX + python-louvain (graph analysis)

## Project Structure
fraudlens/
├── data/        ← Elliptic dataset (not tracked)
├── src/         ← core pipeline code
├── models/      ← saved model weights (not tracked)
├── notebooks/   ← exploratory analysis
└── dashboard/   ← Streamlit app

## Dataset
Elliptic Bitcoin Transaction Dataset — [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
EOF