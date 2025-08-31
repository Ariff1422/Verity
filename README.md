# Verity — Customer Review Sieve (TikTok TechJam 2025)

Verity is a machine learning project designed to filter and highlight well-intended customer reviews—ensuring that genuine, constructive feedback reaches businesses. Built for the TikTok TechJam 2025, Verity guides businesses toward meaningful improvements while minimizing noise from irrelevant or spammy reviews.

---

## 📂 Repository Structure

```
Verity/
 ├── data/             # Raw and processed datasets
 ├── notebooks/        # Exploratory data analysis & preprocessing notebooks and final results under testing.ipynb
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

Navigate to the /src directory
Then run:
```bash
python app.py
```

\* adjust based on your chosen framework.

### 2. Data Preparation (some datasets have been left out due to GitHub size issues)

- **Step 1:** Place raw review data into `data/raw/`.
- **Step 2:** Run preprocessing notebooks in `notebooks/` to:
  - Clean text: remove noise, normalize punctuation/casing
  - Handle missing values
  - Label or categorize reviews (if necessary)

### 3. Augmentation

Augment the processed data to enhance model robustness:
- Apply techniques like synonym replacement, back-translation, random insertion/swap (via `nlpaug` or custom scripts)
- Balance class distribution if skewed (e.g., oversample underrepresented classes)

### 4. Model Training

Workflow:
- Loads data and tokenizes/codes text inputs
- Defines and trains the model
- Saves checkpoints and best-performing model to `models/`
- Logs performance metrics (accuracy, F1, precision, recall)

### 5. Evaluation & Inference
- Validate model on held-out test sets
- Generate evaluation reports and confusion matrices
- Produce predictions on new review data

---

## 🎯 Goals & Motivation

- **Precision-first**: Prioritize the delivery of constructive, high-value customer feedback.
- **Explainable pipeline**: Use data augmentation and notebooks for transparency.
- **Scalable and modular**: Components for data processing, augmentation, training, and inference are decoupled.

---

## 📊 Detailed Process Documentation

### A. Data Wrangling

1. **Raw Data Ingestion**
   - Load data from sources (CSV, JSON, databases) into `data/raw/`--> too big to upload.
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

2. **Balancing Classes**
   - Identify underrepresented classes.
   - Augment these more aggressively or oversample them to achieve near-equal distribution.

3. **Pipeline**
   - Apply augmentation during training or store augmented versions.
   - Use the inital Gradient-Boosting Model to expand labelled set
   - Keep a mapping of original to augmented samples for traceability.

4. **Quality Check**
   - Sample augmented outputs in notebooks to assess semantic fidelity and readability.

---

### C. Model Training

1. **Gradient Boosting (GB) Model**
   - **Architecture**
     - Gradient Boosting model (e.g., XGBoost, LightGBM) used as a baseline.  
     - Trained on structured, vectorized text data.  
     - Key hyperparameters: `learning_rate`, `n_estimators`, `max_depth`, `subsample`.  
   - **Training Workflow**
     - Transform training data using a TF-IDF vectorizer.  
     - Use K-fold cross-validation for robustness.  
     - Tune hyperparameters with GridSearchCV or RandomizedSearchCV.  
     - Save best model in the `models/` directory.  
   - **Logging & Metrics**
     - Track training loss curve.  
     - Metrics: Accuracy, Precision, Recall, F1-score.  
     - Visualize confusion matrix to analyze misclassifications.  
   - **Evaluation**
     - Evaluate final GB model on the held-out test set.  
   - **Inference**
     - Preprocess new text with the trained TF-IDF vectorizer.  
     - Pass vectorized input to GB model for prediction.  

---

2. **Transformer-Based Model**
   - **Architecture**
     - Pre-trained Transformer (e.g., BERT, RoBERTa) fine-tuned for classification.  
     - Simple classification head (linear layer) on top of Transformer outputs.  
     - Key hyperparameters: `learning_rate` (~2×10⁻⁵), `batch_size` (8–32), `epochs`, optimizer = AdamW.  
   - **Training Workflow**
     - Tokenize data using Transformer tokenizer.  
     - Implement training loop with validation split.  
     - Apply early stopping and save best checkpoints (`.ckpt` or `.pth`).  
   - **Logging & Metrics**
     - Track training & validation loss curves.  
     - Metrics: Accuracy, Precision, Recall, F1-score.  
     - Visualize confusion matrix for misclassification analysis.  
   - **Evaluation**
     - Evaluate fine-tuned Transformer on held-out test set.  
   - **Inference**
     - Tokenize new text.  
     - Pass through fine-tuned Transformer for prediction.  

---

3. **Ensemble Model**
   - **Architecture**
     - Voting Classifier combining the best Transformer and GB model.  
     - Supports hard or soft voting for consensus prediction.  
   - **Training Workflow**
     - Fit ensemble on outputs of the base models (no new data learned).  
   - **Logging & Metrics**
     - Metrics: Accuracy, Precision, Recall, F1-score.  
     - Compare ensemble performance against base models.  
   - **Evaluation**
     - Evaluate ensemble on held-out test set.  
   - **Inference**
     - Pass new review through both GB and Transformer models.  
     - Combine predictions with Voting Classifier for final decision.  

## 🤝 Contributing

Your insights or improvements are welcome! To contribute:
1. Fork the repo
2. Create a branch with a clear name (e.g., `feature/augmentation-enhancement`)
3. Submit a PR describing your changes

## 📈 Results

We evaluated **Verity’s ensemble model** on a held-out test set of 800 samples.  
The system achieved an **overall accuracy of 88.5%**, showing that the ensemble consistently outperforms individual models by leveraging their combined strengths.

### 🧪 Evaluation Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **1** | 0.94      | 0.94   | 0.94     | 200     |
| **2** | 0.83      | 0.88   | 0.85     | 202     |
| **3** | 0.98      | 0.86   | 0.92     | 201     |
| **4** | 0.81      | 0.86   | 0.83     | 197     |

**Overall Metrics:**
- **Accuracy:** 0.885  
- **Macro Avg:** Precision 0.89 | Recall 0.88 | F1-Score 0.89  
- **Weighted Avg:** Precision 0.89 | Recall 0.89 | F1-Score 0.89  

---

### 🔎 Key Takeaways
- The ensemble achieved **high precision and recall across all classes**, with especially strong performance in **Class 1 (0.94 F1)** and **Class 3 (0.92 F1)**.  
- **Class 2 and 4** showed slightly lower precision but solid recall, indicating the system is good at identifying them but occasionally misclassifies.  
- The results confirm that the **ensemble approach significantly boosts reliability**, outperforming individual models and producing a balanced performance across all categories.  

---

## 📜 License

Distributed under the MIT License. See [LICENSE](./LICENSE) for details.
