import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

# -------------------------------
# LOAD MODEL
# -------------------------------

print("Loading model...")

brain = joblib.load("final_model.joblib")
embedder = SentenceTransformer("embedder_model")

labels = ["POSITIVE", "NEGATIVE", "NEUTRAL"]

print("System ready\n")

# -------------------------------
# AMBIGUITY THRESHOLDS
# -------------------------------

LOOP_EXTREME      = 14    # extreme topology alone triggers ambiguity
LOOP_MODERATE     = 10    # moderate topology + low gap triggers ambiguity
GAP_THRESHOLD     = 0.30  # confidence gap below this = uncertain classifier

# -------------------------------
# MAIN FUNCTION
# -------------------------------

def predict_with_ambiguity(text):

    # -------------------------------
    # STEP 1: FEATURE EXTRACTION
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
    # STEP 2: PREDICTION
    # -------------------------------
    probs = []
    for i in range(3):
        probs.append(brain["models"][i].predict_proba(v_final)[0][1])

    probs = np.array(probs)

    # -------------------------------
    # STEP 3: CONFIDENCE
    # -------------------------------
    max_prob     = np.max(probs)
    second_prob  = np.sort(probs)[-2]
    confidence_gap = max_prob - second_prob

    # -------------------------------
    # STEP 4: TOPOLOGY
    # -------------------------------
    diagram = diagram_full[0]
    loops = np.sum(diagram[:, 2] == 1)

    # -------------------------------
    # DEBUG
    # -------------------------------
    print("\n--- DEBUG INFO ---")
    print(f"Input          : {text}")
    print(f"Probabilities  : POS={probs[0]:.4f}  NEG={probs[1]:.4f}  NEU={probs[2]:.4f}")
    print(f"Confidence gap : {confidence_gap:.4f}  (threshold < {GAP_THRESHOLD})")
    print(f"Loop count (H1): {loops}  (extreme >= {LOOP_EXTREME}, moderate >= {LOOP_MODERATE})")

    # -------------------------------
    # FINAL DECISION
    # -------------------------------

    # Rule 1: Extreme topology alone = ambiguous
    # Very high loop count means the semantic neighbourhood is highly complex
    # regardless of what the classifier thinks
    if loops >= LOOP_EXTREME:
        print("Trigger        : EXTREME topology")
        return "AMBIGUOUS"

    # Rule 2: Moderate topology + uncertain classifier = ambiguous
    # Both geometric and probabilistic signals agree there is no clear winner
    if loops >= LOOP_MODERATE and confidence_gap < GAP_THRESHOLD:
        print("Trigger        : MODERATE topology + LOW confidence gap")
        return "AMBIGUOUS"

    # Rule 3: Classifier is very uncertain even without strong topology
    # Catches cases where all three probabilities are bunched together
    if confidence_gap < 0.10:
        print("Trigger        : VERY LOW confidence gap (no dominant class)")
        return "AMBIGUOUS"

    # Otherwise: normal prediction
    return labels[np.argmax(probs)]


# -------------------------------
# INTERACTIVE LOOP
# -------------------------------

while True:
    text = input("\nEnter text: ")

    if text.lower() in ["exit", "quit"]:
        break

    result = predict_with_ambiguity(text)
    print("Final Result   :", result)
