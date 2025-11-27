import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib

# Paths
DATA_PATH = "data/mental_health.csv"
MODEL_PATH = "models/crisis_classifier.pkl"

def train_classifier():
    print("⏳ Loading training data...")
    df = pd.read_csv(DATA_PATH)
    
    # NOTE: Adjust these column names based on your specific CSV!
    # Usually Mental Health Corpus has 'text' and 'label' columns
    # We assume 'label' is 1 for toxic/crisis and 0 for normal.
    X = df['text'] 
    y = df['label']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build a pipeline: Convert text to vectors -> Train Classifier
    # We use LogisticRegression because it's fast and effective for simple text classification
    model = make_pipeline(TfidfVectorizer(), LogisticRegression())

    print("🏋️ Training the safety model...")
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Safety model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_classifier()