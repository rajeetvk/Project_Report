import os
os.environ["HF_DATASETS_OFFLINE"] = "1"

import numpy as np
import pandas as pd
import joblib
import torch
import warnings

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceLandscape

from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# -------------------------------
# device setup
# -------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

MODEL_FILE = "final_model.joblib"
EMBEDDER_PATH = "embedder_model"
DATASET_PATH = "./dataset"

# -------------------------------
# label mapping
# -------------------------------

mapping_binary = {
    "positive": [0,1,4,5,13,15,18,20,21,24],
    "negative": [2,3,6,7,9,10,11,12,14,16,19,22,23,25,26],
    "neutral": [27]
}

emotion_names = ["POSITIVE", "NEGATIVE", "NEUTRAL"]

# -------------------------------
# corrected topology function
# -------------------------------

def compute_topology(i, X_query, X_reference, nbrs, VR, PL):

    _, indices = nbrs.kneighbors(X_query[i].reshape(1, -1))

    # always use training embeddings as reference
    cloud = X_reference[indices[0]].reshape(1, 24, -1)

    diag = VR.fit_transform(cloud)
    land = PL.fit_transform(diag).flatten()

    return land

# -------------------------------
# build model
# -------------------------------

def build_model():

    print("Loading dataset...")
    dataset = load_dataset("go_emotions", "simplified", cache_dir=DATASET_PATH)

    df_train = pd.DataFrame(dataset["train"])
    df_test = pd.DataFrame(dataset["test"])

    print("Training samples:", len(df_train))

    # labels
    y_train = np.array(df_train["labels"].apply(
        lambda x: [1 if any(i in x for i in cat) else 0 for cat in mapping_binary.values()]
    ).tolist())

    y_test = np.array(df_test["labels"].apply(
        lambda x: [1 if any(i in x for i in cat) else 0 for cat in mapping_binary.values()]
    ).tolist())

    # -------------------------------
    # embedding (GPU)
    # -------------------------------

    if os.path.exists(EMBEDDER_PATH):
        embedder = SentenceTransformer(EMBEDDER_PATH, device=DEVICE)
    else:
        embedder = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
        embedder.save(EMBEDDER_PATH)

    print("Encoding training data...")
    X_dense_train = embedder.encode(
        df_train["text"].tolist(),
        batch_size=256,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    print("Encoding test data...")
    X_dense_test = embedder.encode(
        df_test["text"].tolist(),
        batch_size=256,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    # -------------------------------
    # tfidf
    # -------------------------------

    print("Creating TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1,2), stop_words="english")

    X_tfidf_train = tfidf.fit_transform(df_train["text"]).toarray()
    X_tfidf_test = tfidf.transform(df_test["text"]).toarray()

    # -------------------------------
    # neighbors
    # -------------------------------

    print("Building neighbor graph...")
    nbrs = NearestNeighbors(n_neighbors=24, metric="cosine", n_jobs=-1)
    nbrs.fit(X_dense_train)

    # -------------------------------
    # topology
    # -------------------------------

    print("Computing topology (this will take time)...")

    VR = VietorisRipsPersistence(homology_dimensions=[0,1], n_jobs=-1)
    PL = PersistenceLandscape(n_layers=2, n_bins=30)

    # TRAIN topology
    topo_train = Parallel(n_jobs=-1)(
        delayed(compute_topology)(i, X_dense_train, X_dense_train, nbrs, VR, PL)
        for i in range(len(X_dense_train))
    )

    # TEST topology (IMPORTANT FIX)
    topo_test = Parallel(n_jobs=-1)(
        delayed(compute_topology)(i, X_dense_test, X_dense_train, nbrs, VR, PL)
        for i in range(len(X_dense_test))
    )

    X_topo_train = np.array(topo_train)
    X_topo_test = np.array(topo_test)

    # -------------------------------
    # combine features
    # -------------------------------

    X_train = np.hstack([X_dense_train, X_tfidf_train, X_topo_train])
    X_test = np.hstack([X_dense_test, X_tfidf_test, X_topo_test])

    # -------------------------------
    # train models
    # -------------------------------

    print("Training models...")

    models = []
    for i in range(3):
        print("Training model", i + 1)
        model = XGBClassifier(n_estimators=100, max_depth=6, tree_method="hist")
        model.fit(X_train, y_train[:, i])
        models.append(model)

    # -------------------------------
    # evaluation
    # -------------------------------

    print("Evaluating model...")

    preds = []
    for i in range(3):
        preds.append(models[i].predict(X_test))

    preds = np.array(preds).T

    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(preds, axis=1)

    acc = accuracy_score(y_true, y_pred)

    print("\nFinal Accuracy:", acc)
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=emotion_names))

    return {
        "models": models,
        "tfidf": tfidf,
        "nbrs": nbrs,
        "X_dense_train": X_dense_train,
        "VR": VR,
        "PL": PL
    }

# -------------------------------
# load or train
# -------------------------------

if os.path.exists(MODEL_FILE):
    print("Loading saved model...")
    brain = joblib.load(MODEL_FILE)
    embedder = SentenceTransformer(EMBEDDER_PATH, device=DEVICE)
else:
    brain = build_model()
    joblib.dump(brain, MODEL_FILE)
    embedder = SentenceTransformer(EMBEDDER_PATH, device=DEVICE)
    print("Model saved")

# -------------------------------
# interactive mode
# -------------------------------

print("\nSystem ready\n")

while True:
    text = input("Enter text: ")

    if text.lower() in ["exit", "quit"]:
        break

    v_dense = embedder.encode([text], convert_to_numpy=True)
    v_tfidf = brain["tfidf"].transform([text]).toarray()

    _, indices = brain["nbrs"].kneighbors(v_dense)
    cloud = brain["X_dense_train"][indices[0]].reshape(1, 24, -1)

    v_diag = brain["VR"].fit_transform(cloud)
    v_topo = brain["PL"].fit_transform(v_diag).reshape(1, -1)

    v_final = np.hstack([v_dense, v_tfidf, v_topo])

    probs = []
    for i in range(3):
        probs.append(brain["models"][i].predict_proba(v_final)[0][1])

    print("Prediction:", emotion_names[np.argmax(probs)])