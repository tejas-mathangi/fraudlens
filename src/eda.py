import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'elliptic_bitcoin_dataset')

FEATURES_PATH = os.path.join(DATA_DIR, 'elliptic_txs_features.csv')
EDGES_PATH    = os.path.join(DATA_DIR, 'elliptic_txs_edgelist.csv')
CLASSES_PATH  = os.path.join(DATA_DIR, 'elliptic_txs_classes.csv')

# ── Load Data ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("FRAUDLENS — Exploratory Data Analysis")
print("=" * 60)

# Features: first column is txId, second is time step, rest are features
features = pd.read_csv(FEATURES_PATH, header=None)
edges    = pd.read_csv(EDGES_PATH)
classes  = pd.read_csv(CLASSES_PATH)

print(f"\n[1] RAW SHAPES")
print(f"    Features : {features.shape}  (rows=transactions, cols=txId + timestep + 166 features)")
print(f"    Edges    : {edges.shape}  (rows=edges between transactions)")
print(f"    Classes  : {classes.shape}  (rows=transactions, cols=txId + label)")

# ── Feature DataFrame cleanup ──────────────────────────────────────────────────
# Column 0 = transaction ID, Column 1 = time step, Columns 2-167 = features
features.columns = ['txId', 'time_step'] + [f'f{i}' for i in range(1, 166)]

print(f"\n[2] TIME STEPS")
print(f"    Total time steps : {features['time_step'].nunique()}")
print(f"    Range            : {features['time_step'].min()} → {features['time_step'].max()}")

# ── Class Distribution ─────────────────────────────────────────────────────────
print(f"\n[3] CLASS DISTRIBUTION")
class_counts = classes['class'].value_counts()
print(class_counts.to_string())

# Map to readable labels
label_map = {'1': 'illicit', '2': 'licit', 'unknown': 'unknown'}
classes['label'] = classes['class'].astype(str).map(label_map)

labeled   = classes[classes['label'] != 'unknown']
illicit   = classes[classes['label'] == 'illicit']
licit     = classes[classes['label'] == 'licit']
unknown   = classes[classes['label'] == 'unknown']

total = len(classes)
print(f"\n    Total transactions : {total:,}")
print(f"    Illicit (fraud)    : {len(illicit):,}  ({100*len(illicit)/total:.1f}%)")
print(f"    Licit (clean)      : {len(licit):,}  ({100*len(licit)/total:.1f}%)")
print(f"    Unknown            : {len(unknown):,}  ({100*len(unknown)/total:.1f}%)")
print(f"\n    Among LABELED only:")
print(f"    Illicit            : {100*len(illicit)/len(labeled):.1f}%")
print(f"    Licit              : {100*len(licit)/len(labeled):.1f}%")
print(f"    → Class imbalance ratio: 1:{len(licit)//len(illicit)}")

# ── Edge Stats ─────────────────────────────────────────────────────────────────
print(f"\n[4] GRAPH STRUCTURE")
print(f"    Total edges        : {len(edges):,}")
print(f"    Unique source nodes: {edges.iloc[:,0].nunique():,}")
print(f"    Unique target nodes: {edges.iloc[:,1].nunique():,}")

# ── Feature Stats ──────────────────────────────────────────────────────────────
feature_cols = [f'f{i}' for i in range(1, 166)]
print(f"\n[5] FEATURE STATISTICS (first 5 features)")
print(features[feature_cols[:5]].describe().round(3).to_string())

print(f"\n    Missing values in features: {features[feature_cols].isnull().sum().sum()}")
print(f"    Missing values in classes : {classes['class'].isnull().sum()}")

# ── Plots ──────────────────────────────────────────────────────────────────────
print(f"\n[6] Generating plots...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('FraudLens — Elliptic Dataset EDA', fontsize=14, fontweight='bold')

# Plot 1: Class distribution
ax1 = axes[0]
labels_plot = ['Illicit', 'Licit', 'Unknown']
sizes  = [len(illicit), len(licit), len(unknown)]
colors = ['#e74c3c', '#2ecc71', '#95a5a6']
ax1.bar(labels_plot, sizes, color=colors, edgecolor='black', linewidth=0.5)
ax1.set_title('Transaction Class Distribution')
ax1.set_ylabel('Count')
for i, v in enumerate(sizes):
    ax1.text(i, v + 200, f'{v:,}', ha='center', fontsize=9)

# Plot 2: Illicit transactions per time step
ax2 = axes[1]
merged = features[['txId', 'time_step']].merge(classes[['txId', 'label']], on='txId')
illicit_per_ts = merged[merged['label'] == 'illicit'].groupby('time_step').size()
licit_per_ts   = merged[merged['label'] == 'licit'].groupby('time_step').size()
ax2.plot(illicit_per_ts.index, illicit_per_ts.values, color='#e74c3c', label='Illicit', linewidth=2)
ax2.plot(licit_per_ts.index,   licit_per_ts.values,   color='#2ecc71', label='Licit',   linewidth=2)
ax2.set_title('Transactions per Time Step')
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Count')
ax2.legend()

# Plot 3: Feature correlation heatmap (first 20 features)
ax3 = axes[2]
sample_features = features[feature_cols[:20]].sample(1000, random_state=42)
corr = sample_features.corr()
sns.heatmap(corr, ax=ax3, cmap='coolwarm', center=0,
            xticklabels=False, yticklabels=False, cbar=True)
ax3.set_title('Feature Correlation (first 20 features)')

plt.tight_layout()
plot_path = os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'eda_plots.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"    Plots saved to notebooks/eda_plots.png")

print("\n" + "=" * 60)
print("EDA COMPLETE")
print("=" * 60)