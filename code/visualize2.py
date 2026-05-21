import joblib
import numpy as np
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from gtda.plotting import plot_diagram

# load trained model
brain = joblib.load("final_model.joblib")
embedder = SentenceTransformer("embedder_model")

print("Model loaded")

# -----------------------------------
# FUNCTION TO VISUALIZE ONE SENTENCE
# -----------------------------------

def visualize_text(text, name):

    print(f"\nProcessing: {text}")

    # encode input
    v_dense = embedder.encode([text])

    # find neighbors from training
    _, indices = brain["nbrs"].kneighbors(v_dense)
    neighbors = brain["X_dense_train"][indices[0]]

    # -------------------------------
    # POINT CLOUD
    # -------------------------------

    pca = PCA(n_components=2)
    points_2d = pca.fit_transform(neighbors)

    plt.figure()
    plt.scatter(points_2d[:, 0], points_2d[:, 1])
    plt.title(f"Point Cloud - {name}")
    plt.savefig(f"{name}_cloud.png")
    plt.close()

    # -------------------------------
    # TOPOLOGY
    # -------------------------------

    cloud = neighbors.reshape(1, 24, -1)

    diagram_full = brain["VR"].fit_transform(cloud)
    diagram = diagram_full[0]

    fig = plot_diagram(diagram)
    fig.write_image(f"{name}_diagram.png")

    # -------------------------------
    # LANDSCAPE
    # -------------------------------

    landscape = brain["PL"].fit_transform(diagram_full)

    plt.figure()
    plt.plot(landscape.flatten())
    plt.title(f"Landscape - {name}")
    plt.savefig(f"{name}_landscape.png")
    plt.close()

    print(f"Saved images for {name}")


# -----------------------------------
# TEST CASES
# -----------------------------------

# clear emotion (strong cluster)
text1 = "I am very happy and excited"

# ambiguous / mixed
text2 = "I am fine"

visualize_text(text1, "clear")
visualize_text(text2, "ambiguous")

print("\nDone")