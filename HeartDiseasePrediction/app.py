import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- 1. MODEL TRAINING AND SETUP ---
DATA_PATH = 'heart_disease_cleaned.csv'

# Load Data
try:
    data = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: Dataset not found at {DATA_PATH}. Please ensure it is in the same directory.")
    exit()

# Clean up column names by lowercasing
data.columns = data.columns.str.lower()
data = data.rename(columns={'num': 'target'}) # 'num' is the target variable (0 = No Disease, 1-4 = Disease)

# Simplify target variable: 0 for no disease, 1 for presence of disease
data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)

# Drop irrelevant dataset column
data = data.drop('dataset', axis=1)
# Drop ID column
data = data.drop('id', axis=1)

# Separate features and target
X = data.drop('target', axis=1)
y = data['target']

# Define feature types
numerical_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)

# Create the full pipeline (Preprocessor + Model)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
print("Training Random Forest Classifier...")
model.fit(X, y)
print("Model training complete.")

# --- 2. FLASK SERVER SETUP ---
app = Flask(__name__)
# Enable CORS for the client (HTML file) running on a different port/origin
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data sent from the HTML client (it's a JSON object)
        raw_data = request.get_json(force=True)

        # Convert the incoming JSON data into a pandas DataFrame row
        # We need the columns to be in the exact order the model expects (X.columns)
        input_data = pd.DataFrame([raw_data], columns=X.columns)

        # Perform prediction
        # model.predict returns 0 (No Disease) or 1 (Disease)
        prediction_class = model.predict(input_data)[0]

        # Return only the prediction class to the client, removing probability and confidence
        return jsonify({
            'prediction': int(prediction_class)
        })

    except Exception as e:
        # Catch any errors during processing and return an error message
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)}), 400

# --- 3. RUN SERVER ---
if __name__ == '__main__':
    app.run(debug=True)
