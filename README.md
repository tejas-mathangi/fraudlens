# FraudLens

A Graph Neural Network based fraud detection system built on the Elliptic Bitcoin Dataset.

## 🎯 Overview

FraudLens detects coordinated financial fraud using **GraphSAGE**, community detection, and **GNNExplainer** to identify fraudulent accounts, fraud rings, and explain suspicious transaction patterns. This system leverages graph neural networks to capture the relationships between Bitcoin addresses and detect anomalous behavior patterns that indicate coordinated fraud.

## ✨ Key Features

- **Graph Neural Networks**: GraphSAGE-based architecture for transaction network analysis
- **Fraud Ring Detection**: Community detection algorithms to identify organized fraud groups
- **Explainability**: GNNExplainer integration to understand model predictions
- **Interactive Dashboard**: Real-time visualization and monitoring with Streamlit
- **Scalable Design**: Efficient processing of large transaction networks

## 🛠️ Tech Stack

- **Python 3.12** — Core language
- **PyTorch 2.2.2** + **PyTorch Geometric 2.7.0** — Deep learning framework
- **Streamlit** — Interactive dashboard and UI
- **NetworkX** + **python-louvain** — Graph analysis and community detection
- **Scikit-learn** — Machine learning utilities

## 📁 Project Structure

```
fraudlens/
├── data/            # Elliptic dataset (not tracked)
├── src/             # Core pipeline code
│   ├── models/      # GNN model implementations
│   ├── preprocessing/ # Data processing utilities
│   └── utils/       # Helper functions
├── models/          # Saved model weights (not tracked)
├── notebooks/       # Exploratory analysis and experiments
├── dashboard/       # Streamlit application
├── tests/           # Unit tests
├── requirements.txt # Project dependencies
└── README.md        # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- pip or conda

### Installation

1. Clone the repository:
```bash
git clone https://github.com/tejas-mathangi/fraudlens.git
cd fraudlens
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the Elliptic dataset:
   - Visit [Kaggle Elliptic Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
   - Extract to `data/` directory

### Running the Dashboard

```bash
streamlit run dashboard/app.py
```

## 📊 Dataset

**Elliptic Bitcoin Transaction Dataset** — [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

- 200K+ Bitcoin transactions
- 49 features per transaction
- Labeled fraud/licit transactions
- Real-world temporal dynamics

## 🔬 Model Architecture

- **Graph Representation**: Transaction networks as heterogeneous graphs
- **Aggregation**: GraphSAGE mean/LSTM aggregators
- **Layers**: 2-3 graph convolutional layers
- **Explainability**: GNNExplainer for feature importance

## 📈 Results

[Add your key metrics, accuracy, precision, recall, or other relevant results here]

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests. For major changes, please open an issue first to discuss proposed changes.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a pull request

## 📝 License

This project is licensed under the MIT License — see the LICENSE file for details.

## 📧 Contact & Support

- **Author**: [Your Name/Profile]
- **Issues**: [GitHub Issues](https://github.com/tejas-mathangi/fraudlens/issues)
- **Discussions**: [GitHub Discussions](https://github.com/tejas-mathangi/fraudlens/discussions)

## 🙏 Acknowledgments

- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) for GNN implementations
- [Elliptic](https://www.elliptic.co/) for the Bitcoin dataset
- [Streamlit](https://streamlit.io/) for the dashboard framework

---

**Note**: This project is for educational and research purposes. Always ensure compliance with applicable regulations when working with financial data.
