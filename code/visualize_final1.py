import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

from sklearn.decomposition import PCA
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceLandscape

# -------------------------------
# SETTINGS
# -------------------------------
warnings.filterwarnings("ignore")
os.makedirs("save", exist_ok=True)

plt.rcParams.update({
    "font.size": 12,
    "figure.titlesize": 16,
    "axes.titlesize": 14
})

# -------------------------------
# LOAD MODEL
# -------------------------------
print("Loading model...")
brain = joblib.load("final_model.joblib")
X = brain["X_dense_train"]

# -------------------------------
# SAMPLE DATA
# -------------------------------
print("Sampling data...")
np.random.seed(42)

sample_size = 80
sample_idx = np.random.choice(len(X), sample_size, replace=False)
sample = X[sample_idx]

# reshape for TDA
sample_reshaped = sample.reshape(1, sample.shape[0], sample.shape[1])

# -------------------------------
# PCA (for visualization)
# -------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(sample)

# ===============================
# 🔵 1. POINT CLOUD
# ===============================
plt.figure(figsize=(7,6))

sns.kdeplot(
    x=X_pca[:,0],
    y=X_pca[:,1],
    fill=True,
    cmap="viridis",
    levels=25,
    alpha=0.6
)

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    s=30,
    color='black',
    alpha=0.8,
    edgecolor='white',
    linewidth=0.3
)

plt.title("Point Cloud Visualization in Embedding Space")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True, linestyle='--', alpha=0.4)

plt.savefig("save/point_cloud_general.png", dpi=400, bbox_inches='tight')
plt.close()

# ===============================
# 🔴 2. PERSISTENCE DIAGRAM
# ===============================
print("Computing persistence diagram...")

VR = VietorisRipsPersistence(homology_dimensions=[0,1])
diagram = VR.fit_transform(sample_reshaped)[0]

plt.figure(figsize=(6,6))

# Separate H0 and H1
h0 = diagram[diagram[:,2] == 0]
h1 = diagram[diagram[:,2] == 1]

plt.scatter(h0[:,0], h0[:,1], color='red', label='H0 (components)', s=40)
plt.scatter(h1[:,0], h1[:,1], color='green', label='H1 (loops)', s=40)

# Diagonal
max_val = np.max(diagram[:,1])
plt.plot([0, max_val], [0, max_val], 'k--', linewidth=1)

plt.title("Persistence Diagram")
plt.xlabel("Birth")
plt.ylabel("Death")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)

plt.savefig("save/persistence_diagram.png", dpi=400, bbox_inches='tight')
plt.close()

# ===============================
# 🟢 3. PERSISTENCE LANDSCAPE
# ===============================
print("Computing persistence landscape...")

PL = PersistenceLandscape(n_layers=3, n_bins=50)
landscape = PL.fit_transform(diagram.reshape(1, -1, 3))[0]

plt.figure(figsize=(7,5))

for i, layer in enumerate(landscape):
    plt.plot(layer, label=f"Layer {i+1}")

plt.title("Persistence Landscape")
plt.xlabel("Filtration Scale (discretized)")  # ✅ FIXED LABEL
plt.ylabel("Landscape Function")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)

plt.savefig("save/persistence_landscape.png", dpi=400, bbox_inches='tight')
plt.close()

# -------------------------------
# DONE
# -------------------------------
print("\n✅ FINAL images saved in /save folder")