import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

print("Loading model...")

brain = joblib.load("final_model.joblib")
embedder = SentenceTransformer("embedder_model")

print("Loading dataset...")

dataset = load_dataset("go_emotions", "simplified")
test_data = dataset["test"]

texts = test_data["text"]
labels_raw = test_data["labels"]

# -------------------------------
# LABEL MAPPING (FIXED)
# -------------------------------

def map_labels(label_list):
    if any(l in label_list for l in [0,13,15,18,21,24]):
        return "POSITIVE"
    elif any(l in label_list for l in [2,3,11,14]):
        return "NEGATIVE"
    else:
        return "NEUTRAL"

y_true = np.array([map_labels(l) for l in labels_raw])

print("Encoding text...")

X_dense = embedder.encode(texts)
X_tfidf = brain["tfidf"].transform(texts).toarray()

print("Finding neighbors...")

_, indices = brain["nbrs"].kneighbors(X_dense)
neighbors = brain["X_dense_train"][indices]

print("Computing topology (this may take time)...")

topo_features = []

for i in range(len(X_dense)):
    cloud = neighbors[i].reshape(1, 24, -1)
    diagram = brain["VR"].fit_transform(cloud)
    landscape = brain["PL"].fit_transform(diagram).flatten()
    topo_features.append(landscape)

X_topo = np.array(topo_features)

print("Combining features...")

X_final = np.hstack([X_dense, X_tfidf, X_topo])

print("Predicting...")

probs = []
for i in range(3):
    probs.append(brain["models"][i].predict_proba(X_final)[:,1])

probs = np.array(probs).T

label_names = ["POSITIVE", "NEGATIVE", "NEUTRAL"]

# Correct mapping
y_pred = [label_names[i] for i in np.argmax(probs, axis=1)]

# -------------------------------
# METRICS
# -------------------------------

print("\n===== RESULTS =====")

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='weighted')
rec = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")

# -------------------------------
# CORRECT CLASSIFICATION REPORT
# -------------------------------

print("\nClassification Report:\n")

print(classification_report(
    y_true,
    y_pred,
    labels=label_names,
    target_names=label_names
))