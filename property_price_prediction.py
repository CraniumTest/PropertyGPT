import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import joblib
from flask import Flask, jsonify, request

def load_and_preprocess_data(filepath):
    # Load dataset
    data = pd.read_csv(filepath)
    # Simple preprocessing
    data.dropna(inplace=True)
    features = data.drop('price', axis=1)
    target = data['price']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model = GradientBoostingRegressor()
    model.fit(X_train_scaled, y_train)
    return model, scaler

def evaluate_model(model, X_test, y_test, scaler):
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"RMSE: {rmse}")

def save_model(model, scaler, filepath):
    joblib.dump((model, scaler), filepath)

def load_model(filepath):
    return joblib.load(filepath)

# Flask API setup
app = Flask(__name__)
model, scaler = load_model('property_price_model.pkl')

@app.route('/predict', methods=['POST'])
def predict_price():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return jsonify({'predicted_price': float(prediction[0])})

if __name__ == "__main__":
    # Load and preprocess your dataset
    X_train, X_test, y_train, y_test = load_and_preprocess_data('property_data.csv')
    
    # Train the model and evaluate
    model, scaler = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, scaler)
    
    # Save the model for later use
    save_model(model, scaler, 'property_price_model.pkl')
    
    # Run the API
    app.run(debug=True)
