# FraudLens 🔍

> Graph Neural Network Based Fraud Detection on the Elliptic Bitcoin Dataset

FraudLens detects coordinated financial fraud using GraphSAGE — a graph neural network that learns from both individual transaction behavior and the relational structure between accounts. Unlike traditional ML approaches that analyze transactions in isolation, FraudLens models the full transaction network and identifies fraud at three levels: individual accounts, organized fraud rings, and structural explanations.

---

## Why Graph-Based?

Traditional fraud detection (Random Forests, XGBoost) treats each transaction as an independent row in a table. Modern fraud is **coordinated** — fraud rings operate as networks of accounts that move money between each other to launder it. A single transaction in a ring may look clean. The network pattern reveals the fraud.

FraudLens represents transactions as a graph:
- **Nodes** = Bitcoin wallet addresses
- **Edges** = transactions between wallets

The GNN learns that an honest wallet surrounded by fraudulent wallets is itself suspicious — something no tabular model can capture.

---

## Results

| Model | AUC-ROC | F1 (Illicit) | Avg Precision | Fraud Rings | Explainability |
|-------|---------|-------------|---------------|-------------|----------------|
| Random Forest (Baseline) | 0.9958 | 0.9316 | 0.9807 | ✗ | ✗ |
| **GraphSAGE (FraudLens)** | **0.9885** | **0.8833** | **0.9566** | **✓** | **✓** |

**10 fraud rings detected** — including communities with 100% illicit concentration and average fraud scores above 0.94.

The Random Forest's higher raw F1 reflects Elliptic's pre-engineered neighborhood features. GraphSAGE learns from raw graph topology without hand-crafted relational features, and uniquely provides fraud ring detection and node-level explainability — capabilities that are architecturally impossible in tabular models.

---

## System Architecture

```
Elliptic Dataset (CSV)
        ↓
Graph Construction
(203,769 nodes · 234,355 edges · 165 features)
        ↓
GraphSAGE Training
(3 layers · weighted loss · 500 epochs)
        ↓
   ┌────┴──────────────────┬─────────────────────┐
   ↓                       ↓                     ↓
Fraud Scoring         Fraud Ring             Explainability
per node              Detection              (gradient-based)
(AUC: 0.9885)        (Louvain clustering)   (feature + edge importance)
   └────────────────────────┴─────────────────────┘
                            ↓
                 Streamlit Dashboard
```

---

## Project Structure

```
fraudlens/
├── src/
│   ├── graph_builder.py   ← loads CSVs, builds PyG Data object
│   ├── model.py           ← GraphSAGE architecture (3 SAGEConv layers)
│   ├── train.py           ← training loop, evaluation, class weighting
│   ├── train_more.py      ← extended training from checkpoint
│   ├── baseline.py        ← Random Forest baseline
│   ├── fraud_rings.py     ← Louvain community detection
│   ├── explainer.py       ← gradient-based GNN explainability
│   └── eda.py             ← exploratory data analysis
├── dashboard/
│   └── app.py             ← Streamlit dashboard (4 pages)
├── notebooks/
│   ├── eda_plots.png
│   ├── baseline_plots.png
│   ├── training_plots.png
│   ├── fraud_rings.png
│   └── explanations.png
├── models/                ← saved model weights (not tracked)
├── data/                  ← Elliptic dataset (not tracked)
└── README.md
```

---

## Stack

| Component | Technology |
|-----------|-----------|
| GNN Framework | PyTorch Geometric 2.7.0 |
| Deep Learning | PyTorch 2.2.2 |
| Graph Analysis | NetworkX + python-louvain |
| Baseline | scikit-learn RandomForest |
| Dashboard | Streamlit |
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn |

---

## Dataset

**Elliptic Bitcoin Transaction Dataset** — [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

| Property | Value |
|----------|-------|
| Nodes (transactions) | 203,769 |
| Edges | 234,355 |
| Features per node | 165 |
| Time steps | 49 |
| Illicit nodes | 4,545 (2.2%) |
| Licit nodes | 42,019 (20.6%) |
| Unknown nodes | 157,205 (77.1%) |

The dataset is not included in this repo (Kaggle license). Download it and place the three CSVs in `data/elliptic_bitcoin_dataset/`.

---

## Setup

```bash
# Clone
git clone https://github.com/tejas-mathangi/fraudlens.git
cd fraudlens

# Create virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install torch==2.2.2 torchvision torchaudio
pip install torch_geometric==2.7.0
pip install pandas numpy scikit-learn matplotlib seaborn
pip install networkx python-louvain streamlit
pip install "numpy<2"
```

---

## Usage

```bash
# 1. Explore the dataset
python3 src/eda.py

# 2. Run baseline (Random Forest)
python3 src/baseline.py

# 3. Train GraphSAGE
python3 src/train.py

# 4. Extended training from checkpoint
python3 src/train_more.py

# 5. Detect fraud rings
python3 src/fraud_rings.py

# 6. Run explainer
python3 src/explainer.py

# 7. Launch dashboard
streamlit run dashboard/app.py
```

---

## Key Design Decisions

**GraphSAGE over GCN** — Inductive learning means the model generalizes to unseen nodes. GCN is transductive and retrains from scratch on new data. For a fraud detection system that sees new transactions daily, inductive capability matters.

**Weighted loss (illicit weight: 5.12×)** — The dataset is 1:9 illicit:licit among labeled nodes. Without class weighting, the model achieves 90% accuracy by predicting everything as licit — useless for fraud detection. Weighted cross-entropy forces the model to treat missed fraud as 5× more costly than a false alarm.

**Louvain on graph topology, not embeddings** — Community detection runs on the raw transaction graph rather than the embedding space. This ensures fraud rings reflect actual transaction relationships, not learned similarity. The GNN fraud scores then validate which communities are high-risk.

**Gradient-based explainability** — For each flagged node, we compute the gradient of the fraud prediction with respect to input features. High-gradient features are the ones the model is most sensitive to — the explanation is faithful to the actual computation, not a post-hoc approximation.

---

## Visualizations

### Fraud Ring Detection
Communities where >40% of labeled nodes are confirmed illicit, with avg fraud score >0.4. Community 206: 97.8% illicit, 45 confirmed fraud nodes, avg score 0.957.

### GNN Explainability
Subgraph visualization showing which neighbors and transaction features drove the fraud prediction for each flagged node.

---

## Academic Context

This project is a research-oriented implementation for a Real-Time Research Project course (B.Tech CSE AI/ML, JNTUH). The methodology is grounded in published literature:

- Hamilton et al. (2017) — GraphSAGE
- Weber et al. (2019) — GNN-based fraud detection on Elliptic
- Ying et al. (2019) — GNNExplainer
- Blondel et al. (2008) — Louvain community detection

---

## Author

**Tejas Mathangi** — B.Tech CSE (AI & ML), JNTUH Hyderabad

[GitHub](https://github.com/tejas-mathangi) · [LinkedIn](https://linkedin.com/in/tejas-mathangi)
