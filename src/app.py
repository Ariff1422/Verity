import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import torch
import joblib
import os
import traceback
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier

# --- Global variables for models ---
models_loaded = False
tinybert_tokenizer = None
tinybert_model = None
gb_model = None
gb_vectorizer = None
ensemble_model = None
error_message = None

# --- Path to your models ---
MODELS_BASE_PATH = '../models'
TINYBERT_PATH = os.path.join(MODELS_BASE_PATH, 'tinybertfinetuned')
GB_MODEL_PATH = os.path.join(MODELS_BASE_PATH, 'gb_model_advanced_tuned.pkl')
VECTORIZER_PATH = os.path.join(MODELS_BASE_PATH, 'vectorizer_advanced_tuned.pkl')
ENSEMBLE_MODEL_PATH = os.path.join(MODELS_BASE_PATH, 'ensemble_meta_model.pkl')

# --- Define your label mapping with integer keys ---
LABEL_MAP = {
    0: "Advertisement",
    1: "Irrelevant Content",
    2: "Rant Without Visit",
    3: "No Violation"
}

# --- Model Loading at Application Startup ---
def load_models():
    """Loads all models from disk."""
    global models_loaded, tinybert_tokenizer, tinybert_model, gb_model, gb_vectorizer, ensemble_model, error_message
    
    try:
        print("--- Loading all models for the Ensemble Classifier ---")
        
        if not os.path.exists(MODELS_BASE_PATH):
            raise FileNotFoundError(f"The 'models' directory was not found at: {MODELS_BASE_PATH}")

        # 1. Load TinyBERT Model
        print("Loading TinyBERT from disk...")
        tinybert_tokenizer = AutoTokenizer.from_pretrained(TINYBERT_PATH)
        tinybert_model = AutoModelForSequenceClassification.from_pretrained(TINYBERT_PATH)
        print("TinyBERT models loaded successfully.")

        # 2. Load your pre-trained GB and vectorizer models
        print("Loading Gradient Boosting and TfidfVectorizer from your files...")
        gb_vectorizer = joblib.load(VECTORIZER_PATH)
        gb_model = joblib.load(GB_MODEL_PATH)
        print("GB model and vectorizer loaded successfully.")
        
        # 3. Load your pre-trained Ensemble Model (Voting Classifier)
        print("Loading your pre-trained Ensemble model...")
        ensemble_model = joblib.load(ENSEMBLE_MODEL_PATH)
        print("Ensemble model loaded successfully.")
        
        models_loaded = True
        print("--- All models loaded and ready. ---")

    except FileNotFoundError as e:
        error_message = f"ERROR: A model file was not found. Please check your file paths. Missing file: {e.filename}"
        print(error_message)
        traceback.print_exc()
    except Exception as e:
        error_message = f"An unexpected error occurred during model loading. Details: {e}"
        print(error_message)
        traceback.print_exc()

# --- The Main Ensemble Classification Function ---
def classify_ensemble(review_text):
    """
    Classifies a review using TinyBERT, a Gradient Boosting model, and an Ensemble model.
    """
    try:
        # --- TinyBERT Prediction ---
        tinybert_input = tinybert_tokenizer(
            review_text,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding='max_length'
        )
        with torch.no_grad():
            tinybert_outputs = tinybert_model(**tinybert_input)
        
        tinybert_probabilities = torch.softmax(tinybert_outputs.logits, dim=1).squeeze().numpy()
        tinybert_prediction = tinybert_outputs.logits.argmax().item()
        tinybert_label = LABEL_MAP.get(tinybert_prediction, "Unknown")

        # --- Gradient Boosting (GB) Prediction ---
        gb_vectorized_input = gb_vectorizer.transform([review_text])
        gb_prediction = gb_model.predict(gb_vectorized_input)[0]
        gb_probabilities = gb_model.predict_proba(gb_vectorized_input)[0]
        gb_label = LABEL_MAP.get(gb_prediction, "Unknown")

        # --- Ensemble Prediction ---
        ensemble_input = np.concatenate([tinybert_probabilities, gb_probabilities]).reshape(1, -1)
        
        ensemble_prediction = ensemble_model.predict(ensemble_input)[0]
        
        ensemble_label = LABEL_MAP.get(int(ensemble_prediction), "Unknown")

        return {
            "TinyBERT Model": tinybert_label,
            "Gradient Boosting Model": gb_label,
            "Ensemble Model": ensemble_label
        }

    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")
        traceback.print_exc()
        return None

# --- Create the Dash application ---
app = dash.Dash(__name__)

# --- Define the app layout ---
app.layout = html.Div(style={'font-family': 'sans-serif', 'padding': '20px', 'max-width': '800px', 'margin': 'auto'}, children=[
    html.H1("Multi-Model Ensemble Review Classifier", style={'textAlign': 'center'}),
    html.P("This app classifies reviews using a TinyBERT, a Gradient Boosting model, and a combined Ensemble model.", style={'textAlign': 'center'}),
    
    html.Div([
        html.H3("Enter a review:"),
        dcc.Textarea(
            id='review-input',
            placeholder="Example: The service was terrible, they need to improve their staff's attitude.",
            style={'width': '100%', 'height': '150px'}
        ),
        html.Button('Classify Review', id='classify-button', n_clicks=0, style={'margin-top': '10px', 'padding': '10px 20px', 'font-size': '16px'}),
    ]),
    
    html.Div(id='results-output', style={'margin-top': '20px'})
])

# --- Define the app callback logic ---
@app.callback(
    Output('results-output', 'children'),
    Input('classify-button', 'n_clicks'),
    State('review-input', 'value')
)
def update_output(n_clicks, review_text):
    if n_clicks > 0:
        if not models_loaded:
            return html.Div(error_message, style={'color': 'red', 'font-weight': 'bold'})
        
        if not review_text:
            return html.Div("Please enter a review to classify.", style={'color': 'orange', 'font-weight': 'bold'})
        
        results = classify_ensemble(review_text)
        
        if results:
            return html.Div(children=[
                html.H3("Classification Results"),
                html.P(f"TinyBERT Model: {results['TinyBERT Model']}"),
                html.P(f"Gradient Boosting Model: {results['Gradient Boosting Model']}"),
                html.P(f"Ensemble Model: {results['Ensemble Model']}", style={'font-weight': 'bold'})
            ])
        else:
            return html.Div("An error occurred during classification. Please check the server console.", style={'color': 'red', 'font-weight': 'bold'})
    return html.Div()

# --- Load models before running the server ---
load_models()

# --- Run the application ---
if __name__ == '__main__':
    app.run(debug=True)
