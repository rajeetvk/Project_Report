import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from gtda.plotting import plot_diagram

# load model
brain = joblib.load("final_model.joblib")

print("Model loaded successfully")

sample_index = 100

# get neighbors
_, indices = brain["nbrs"].kneighbors(
    brain["X_dense_train"][sample_index].reshape(1, -1)
)

neighbors = brain["X_dense_train"][indices[0]]

# -------------------------------
# 1. POINT CLOUD
# -------------------------------

pca = PCA(n_components=2)
points_2d = pca.fit_transform(neighbors)

plt.figure()
plt.scatter(points_2d[:, 0], points_2d[:, 1])
plt.title("Neighbor Structure")
plt.savefig("point_cloud.png")
plt.close()

# -------------------------------
# 2. PERSISTENCE DIAGRAM
# -------------------------------

cloud = neighbors.reshape(1, 24, -1)

diagram_full = brain["VR"].fit_transform(cloud)
diagram = diagram_full[0]

fig = plot_diagram(diagram)
fig.write_image("persistence_diagram.png")

# -------------------------------
# 3. LANDSCAPE
# -------------------------------

landscape = brain["PL"].fit_transform(diagram_full)

plt.figure()
plt.plot(landscape.flatten())
plt.title("Persistence Landscape")
plt.savefig("landscape.png")
plt.close()

print("All plots saved successfully")