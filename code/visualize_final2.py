import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")
os.makedirs("save", exist_ok=True)

# -------------------------------
# LOAD MODEL
# -------------------------------
print("Loading model...")

brain = joblib.load("final_model.joblib")
embedder = SentenceTransformer("embedder_model")

labels = ["POSITIVE", "NEGATIVE", "NEUTRAL"]

print("System ready\n")

# -------------------------------
# THRESHOLDS
# -------------------------------
LOOP_EXTREME  = 14
LOOP_MODERATE = 10
GAP_THRESHOLD = 0.30


# -------------------------------
# CORE FUNCTION
# -------------------------------
def predict_and_plot(text, filename):

    # -------------------------------
    # FEATURE EXTRACTION
    # -------------------------------
    v_dense = embedder.encode([text])
    v_tfidf = brain["tfidf"].transform([text]).toarray()

    _, indices = brain["nbrs"].kneighbors(v_dense)
    neighbors = brain["X_dense_train"][indices[0]]

    cloud = neighbors.reshape(1, 24, -1)
    diagram_full = brain["VR"].fit_transform(cloud)
    v_topo = brain["PL"].fit_transform(diagram_full).reshape(1, -1)

    v_final = np.hstack([v_dense, v_tfidf, v_topo])

    # -------------------------------
    # PREDICTION
    # -------------------------------
    probs = []
    for i in range(3):
        probs.append(brain["models"][i].predict_proba(v_final)[0][1])

    probs = np.array(probs)

    # -------------------------------
    # CONFIDENCE
    # -------------------------------
    max_prob = np.max(probs)
    second_prob = np.sort(probs)[-2]
    gap = max_prob - second_prob

    # -------------------------------
    # TOPOLOGY
    # -------------------------------
    diagram = diagram_full[0]
    loops = np.sum(diagram[:, 2] == 1)

    # -------------------------------
    # DEBUG OUTPUT (same as your report)
    # -------------------------------
    print("\n--- DEBUG INFO ---")
    print(f"Input          : {text}")
    print(f"Probabilities  : POS={probs[0]:.4f}  NEG={probs[1]:.4f}  NEU={probs[2]:.4f}")
    print(f"Confidence gap : {gap:.4f}  (threshold < {GAP_THRESHOLD})")
    print(f"Loop count (H1): {loops}  (extreme >= {LOOP_EXTREME}, moderate >= {LOOP_MODERATE})")

    # -------------------------------
    # DECISION
    # -------------------------------
    if loops >= LOOP_EXTREME:
        print("Trigger        : EXTREME topology")
        result = "AMBIGUOUS"

    elif loops >= LOOP_MODERATE and gap < GAP_THRESHOLD:
        print("Trigger        : MODERATE topology + LOW confidence gap")
        result = "AMBIGUOUS"

    elif gap < 0.10:
        print("Trigger        : VERY LOW confidence gap")
        result = "AMBIGUOUS"

    else:
        result = labels[np.argmax(probs)]

    print("Final Result   :", result)

    # -------------------------------
    # VISUALIZATION
    # -------------------------------
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(neighbors)
    input_pca = pca.transform(v_dense)

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
        alpha=0.8
    )

    plt.scatter(
        input_pca[:,0],
        input_pca[:,1],
        color='red',
        s=180,
        label="Input"
    )

    # ambiguity region
    radius = 0.4 + (loops / LOOP_EXTREME)

    circle = plt.Circle(
        (input_pca[0][0], input_pca[0][1]),
        radius,
        color='red',
        fill=False,
        linestyle='--',
        linewidth=2
    )

    plt.gca().add_patch(circle)

    plt.title(f"Embedding Structure: \"{text}\"")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)

    plt.text(
        input_pca[0][0],
        input_pca[0][1] + 0.3,
        f"{result}\nLoops={loops}, Gap={gap:.2f}",
        fontsize=10,
        ha='center'
    )

    plt.savefig(f"save/{filename}", dpi=400, bbox_inches='tight')
    plt.close()


# -------------------------------
# RUN BOTH CASES
# -------------------------------
predict_and_plot("i am fine", "case1_clear.png")
predict_and_plot("i dont know how i am feel", "case2_ambiguous.png")

print("\n✅ Both case study images saved in /save")