""# MLOps Project: House Price Prediction

# Objective: Predict house prices using historical data and deploy the model with MLOps practices.

# Step 1: Data Management

import pandas as pd
import numpy as np
import os
import sklearn
from sklearn.model_selection import train_test_split

# Load Dataset

data = pd.read_csv('house_prices.csv')
data.ffill(inplace=True)

# Basic Preprocessing

# Only keeping required features
X = data[['bedrooms', 'bathrooms', 'sqft_living']]
y = data['price']

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Data loaded and preprocessed successfully.')


# Step 2: Model Development

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Model Initialization

model = RandomForestRegressor(n_estimators=100, random_state=42)

# Training

model.fit(X_train, y_train)

# Predictions

y_pred = model.predict(X_test)

# Evaluation

mse = mean_squared_error(y_test, y_pred)
print(f'Model Trained. MSE: {mse}')


# Step 3: Model Versioning and Packaging

import joblib
from flask import Flask, request, jsonify

# Save the model

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/house_price_model.pkl')
print('Model saved successfully.')

# Flask App for serving predictions

app = Flask(__name__)

# Home Route

@app.route('/', methods=['GET'])
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>House Price Prediction App</title>
        <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@500&display=swap" rel="stylesheet">
        <style>
            body {
                font-family: 'Montserrat', sans-serif;
                background: linear-gradient(135deg, #74ebd5, #9face6);
                margin: 0;
                padding: 0;
                color: #333;
            }
            .container {
                max-width: 800px;
                margin: 60px auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 0 20px rgba(0,0,0,0.2);
                padding: 40px;
            }
            h2 {
                color: #2c3e50;
                text-align: center;
            }
            ul {
                list-style: none;
                padding-left: 0;
                text-align: center;
            }
            ul li {
                margin: 10px 0;
                font-size: 1.1em;
            }
            form {
                display: flex;
                flex-direction: column;
                gap: 10px;
                margin-top: 20px;
            }
            textarea {
                padding: 10px;
                font-size: 1em;
                border-radius: 8px;
                border: 1px solid #ccc;
                resize: none;
            }
            input[type="submit"] {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 12px;
                font-size: 1em;
                border-radius: 8px;
                cursor: pointer;
                transition: background-color 0.3s;
            }
            input[type="submit"]:hover {
                background-color: #45a049;
            }
            .footer {
                text-align: center;
                margin-top: 30px;
                font-size: 0.9em;
                color: #555;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🏡 House Price Prediction API</h2>
            <p style="text-align:center;">Use the endpoints below to test the model:</p>
            <ul>
                <li><strong>GET /</strong> → Home Page</li>
                <li><strong>POST /predict</strong> → Predict price (send JSON)</li>
            </ul>

            <h3 style="text-align:center;">Try a Prediction:</h3>
            <form action="/predict" method="post" enctype="application/json">
                <label for="features"><strong>Enter Features (JSON format):</strong></label>
                <textarea id="features" name="features" rows="10" cols="50">
{"bedrooms": 3, "bathrooms": 2, "sqft_living": 1800}
                </textarea>
                <input type="submit" value="Predict Price">
            </form>
            <div class="footer">
                <p>Created with ❤️ using Flask & Docker</p>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form['features']
        features = pd.read_json(data, typ='series').to_frame().T
        model = joblib.load('models/house_price_model.pkl')
        prediction = model.predict(features)
        return jsonify({'Prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
""
