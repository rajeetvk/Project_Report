# Emotion Detection Using Topological Data Analysis (TDA)

This folder contains the source code for the Emotion Detection project utilizing Topological Data Analysis (TDA). 

## Setup Instructions

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```
2. **Activate the virtual environment:**
   - **Windows:** `venv\Scripts\activate`
   - **Linux/Mac:** `source venv/bin/activate`

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the model training/inference:**
   ```bash
   python tdanew4.py
   ```
   *Note: On the first run, the model will download the `go_emotions` dataset and train the embedder. The final trained model will be saved locally (it is excluded from version control due to file size constraints).*

## Project Structure
- `tdanew4.py`: Main script to build the model, generate topological features, and train the XGBoost classifier.
- `evaluation.py`: Script used for evaluating the performance and accuracy of the trained model.
- `visualize.py` & variants: Scripts to generate persistence diagrams, landscapes, and point clouds.
- `images/`: Contains structural and diagrammatic exports.
- `save/`: Stores generated case visualisations.
