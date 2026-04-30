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

## Compilation
To generate the final PDF report, compile the main file twice using `pdflatex`:
```bash
pdflatex Final_Project_Report.tex
pdflatex Final_Project_Report.tex
```

## Supervisor
**Dr. Rajkumari Bidyalakshmi Devi**  
Assistant Professor, Dept. of CSE, IIIT Manipur
