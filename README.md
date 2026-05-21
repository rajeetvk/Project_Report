# Emotion Detection from Textual Data with Ambiguity Detection

A B.Tech Project Report by **Arjeet Singh** (230101078), submitted to the Department of Computer Science and Engineering, **Indian Institute of Information Technology (IIIT) Manipur**.

## Overview
This project presents a novel classification pipeline for detecting emotional states from textual data while explicitly quantifying semantic uncertainty. Unlike conventional NLP models that force a classification on every input, this system leverages **Topological Data Analysis (TDA)** to identify "Ambiguous" cases where the semantic content is unclear or sits on the boundary between emotion classes (Positive, Negative, Neutral).

## Key Features
- **Hybrid Feature Extraction:** Combines SentenceTransformers (all-mpnet-base-v2), TF-IDF, and TDA persistence landscapes.
- **Topological Ambiguity Detection:** Uses Vietoris-Rips persistent homology and H1 loop counts to detect geometric complexity in the embedding space.
- **Ensemble Classification:** Utilizes XGBoost models in a one-vs-rest configuration.
- **Principled Abstention:** Implements decision rules based on confidence gaps and topological signals to flag ambiguous inputs.

## Project Structure
The repository is organized for clarity and professional academic standards:
- `Final_Project_Report.tex`: The main LaTeX entry point.
- `chapters/`: Individual chapter source files (1--7).
- `sections/`: Front matter (Abstract, Declaration, Certificate, etc.) and Bibliography.
- `images/`: High-quality figures, diagrams, and TDA plots used in the report.
- `Slides/`: LaTeX Beamer presentation files and associated assets.
- `code/`: Python source code for the TDA-based emotion detection model (see `code/README.md` for setup instructions).

## Example Output
Below is an example of the interaction from running the model (`ambiguity.py`) in the terminal. The model actively calculates confidence gaps and Topological Loop counts (H1) to detect ambiguous data:

```text
Loading model...
Loading weights: 100%|████████████████| 199/199 [00:00<00:00, 8984.09it/s]
System ready

Enter text: I am incredibly happy today

--- DEBUG INFO ---
Input          : I am incredibly happy today
Probabilities  : POS=0.5373  NEG=0.0474  NEU=0.0145
Confidence gap : 0.4899  (threshold < 0.3)
Loop count (H1): 6  (extreme >= 14, moderate >= 10)
Final Result   : POSITIVE

Enter text: I feel so lost and terrible

--- DEBUG INFO ---
Input          : I feel so lost and terrible
Probabilities  : POS=0.2264  NEG=0.7589  NEU=0.0752
Confidence gap : 0.5324  (threshold < 0.3)
Loop count (H1): 12  (extreme >= 14, moderate >= 10)
Final Result   : NEGATIVE

Enter text: i don't know how i feel

--- DEBUG INFO ---
Input          : i don't know how i feel
Probabilities  : POS=0.0738  NEG=0.5692  NEU=0.1989
Confidence gap : 0.3703  (threshold < 0.3)
Loop count (H1): 14  (extreme >= 14, moderate >= 10)
Trigger        : EXTREME topology
Final Result   : AMBIGUOUS

Enter text: i will kill you

--- DEBUG INFO ---
Input          : i will kill you
Probabilities  : POS=0.0625  NEG=0.6266  NEU=0.5617
Confidence gap : 0.0649  (threshold < 0.3)
Loop count (H1): 4  (extreme >= 14, moderate >= 10)
Trigger        : VERY LOW confidence gap (no dominant class)
Final Result   : AMBIGUOUS
```

## Compilation
To generate the final PDF report, compile the main file twice using `pdflatex`:
```bash
pdflatex Final_Project_Report.tex
pdflatex Final_Project_Report.tex
```

## Supervisor
**Dr. Rajkumari Bidyalakshmi Devi**  
Assistant Professor, Dept. of CSE, IIIT Manipur
