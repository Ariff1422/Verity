# Verity — Customer Review Sieve (TikTok TechJam 2025)

Verity is a machine learning project designed to filter and highlight well-intended customer reviews—ensuring that genuine, constructive feedback reaches businesses. Built for the TikTok TechJam 2025, Verity guides businesses toward meaningful improvements while minimizing noise from irrelevant or spammy reviews.

---

## 📂 Repository Structure

```
Verity/
 ├── data/             # Raw and processed datasets
 ├── notebooks/        # Exploratory data analysis & preprocessing notebooks
 ├── src/              # Core modules: data processing, augmentation, training, evaluation
 ├── models/           # Saved trained models and metrics
 ├── README.md         # Project overview and setup guide
 └── LICENSE           # MIT license
```

---

## ⚙️ Setup & Usage

### 1. Environment Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

Ensure your environment includes packages like:
- `pandas`, `numpy`, `scikit-learn`, `torch*` or `tensorflow`
- `nlpaug`, `nltk`, `spaCy` (for augmentation and preprocessing)
- `matplotlib`, `seaborn` (for visualizations)

\* adjust based on your chosen framework.

### 2. Data Preparation

- **Step 1:** Place raw review data into `data/raw/`.
- **Step 2:** Run preprocessing notebooks in `notebooks/` to:
  - Clean text: remove noise, normalize punctuation/casing
  - Handle missing values
  - Label or categorize reviews (if necessary)
  - Save cleaned data to `data/processed/`

### 3. Augmentation

Augment the processed data to enhance model robustness:
- Apply techniques like synonym replacement, back-translation, random insertion/swap (via `nlpaug` or custom scripts)
- Balance class distribution if skewed (e.g., oversample underrepresented classes)
- Store augmented datasets in `data/augmented/`

### 4. Model Training

Run training scripts via:
```bash
python src/train.py \
  --train data/augmented/train.csv \
  --val data/processed/val.csv \
  --output models/ \
  --epochs 10 \
  --batch_size 32 \
  [...other args]
```

This workflow:
- Loads data and tokenizes/codes text inputs
- Defines and trains the model
- Saves checkpoints and best-performing model to `models/`
- Logs performance metrics (accuracy, F1, precision, recall)

### 5. Evaluation & Inference

Use `src/evaluate.py` to:
- Validate model on held-out test sets
- Generate evaluation reports and confusion matrices
- Produce predictions on new review data (save to `predictions/`)

---

## 🎯 Goals & Motivation

- **Precision-first**: Prioritize the delivery of constructive, high-value customer feedback.
- **Explainable pipeline**: Use data augmentation and notebooks for transparency.
- **Scalable and modular**: Components for data processing, augmentation, training, and inference are decoupled.

---

## 📊 Detailed Process Documentation

### A. Data Wrangling

1. **Raw Data Ingestion**
   - Load data from sources (CSV, JSON, databases) into `data/raw/`.
   - Record metadata (e.g., timestamps, source IDs, product IDs).

2. **Cleaning & Preprocessing**
   - Normalize text: lowercase, strip HTML, remove special characters.
   - Tokenize and remove stop words using `spaCy` or `NLTK`.
   - Apply lemmatization/stemming.
   - Optionally extract metadata features (e.g., word counts, sentiment scores).

3. **Labeling**
   - If unlabeled: use heuristics or manual annotation to label reviews as 'helpful' vs. 'not helpful' or similar categories.
   - Save labeled sets for training/validation splitting.

4. **Splitting Dataset**
   - Split into training (70%), validation (15%), and test (15%) sets.
   - Use stratified splits to preserve class balance.

---

### B. Data Augmentation

1. **Techniques**
   - **Synonym Replacement:** Replace words with synonyms (via `WordNet`, `nlpaug`).
   - **Back-Translation:** Translate text to another language and back.
   - **Random Insertion/Swap/Deletion:** Introduce variation while preserving meaning.
   - **Text Perturbation:** Add noise like typos to simulate realistic input.

2. **Balancing Classes**
   - Identify underrepresented classes.
   - Augment these more aggressively or oversample them to achieve near-equal distribution.

3. **Pipeline**
   - Apply augmentation during training or store augmented versions in `data/augmented/`.
   - Keep a mapping of original to augmented samples for traceability.

4. **Quality Check**
   - Sample augmented outputs in notebooks to assess semantic fidelity and readability.

---

### C. Model Training

1. **Architecture**
   - Typical options: Logistic Regression with TF-IDF, LSTM/GRU-based models, Transformer-based classifiers (BERT, RoBERTa).
   - Include hyperparameters: learning rate, batch size, epochs, optimizer, etc.

2. **Training Workflow**
   - Load and vectorize/tokenize data.
   - Implement training loop with validation monitoring.
   - Use early stopping and checkpoints to prevent overfitting.
   - Save best model under `models/`, e.g., `best_model.pth` or `model.ckpt`.

3. **Logging & Metrics**
   - Track training and validation loss curves.
   - Compute performance: Accuracy, Precision, Recall, F1-score.
   - Visualize results with confusion matrix to understand misclassifications.

4. **Evaluation**
   - Evaluate final model on held-out test set.
   - Generate comprehensive reports and save to `models/metrics/`.

5. **Inference**
   - Load trained model.
   - Pass new incoming review data through preprocessing.
   - Output predictions (e.g. “Helpful” vs. “Not Helpful”) in `predictions/`.

---

## 🤝 Contributing

Your insights or improvements are welcome! To contribute:
1. Fork the repo
2. Create a branch with a clear name (e.g., `feature/augmentation-enhancement`)
3. Submit a PR describing your changes

---

## 📜 License

Distributed under the MIT License. See [LICENSE](./LICENSE) for details.
