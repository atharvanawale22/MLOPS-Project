# Advanced House Price Prediction Model with MLOps

# ===============================
#          Import Libraries
# ===============================

import pandas as pd
import numpy as np
import os
import json
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template_string
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from flasgger import Swagger
import shap

# ===============================
#        Data Management
# ===============================

# Load Dataset
data = pd.read_csv('house_prices.csv')
data.ffill(inplace=True)

# Feature Engineering
data['age'] = 2025 - data['yr_built']  # Corrected column name
data['is_renovated'] = data['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)
data['price_per_sqft'] = data['price'] / data['sqft_living']

# Selecting features and target
X = data[['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'condition', 'age', 'is_renovated']]
y = data['price']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
#        Model Development
# ===============================

# Model Initialization
models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Model Training and Evaluation
model_scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_scores[name] = {
        "MSE": mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2_Score": r2_score(y_test, y_pred)
    }

# Select the best model
best_model_name = min(model_scores, key=lambda x: model_scores[x]["MSE"])
best_model = models[best_model_name]

# Save the model
os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/house_price_model.pkl')
print(f"Model '{best_model_name}' saved successfully.")

# ===============================
#        API Development
# ===============================

app = Flask(__name__)
Swagger(app)

# Home Route
@app.route('/', methods=['GET'])
def home():
    return render_template_string('''
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
                margin-bottom: 10px;
            }
            .subtitle {
                text-align: center;
                font-size: 1.1em;
                margin-bottom: 25px;
                color: #555;
            }
            form {
                display: flex;
                flex-direction: column;
                gap: 15px;
                margin-top: 20px;
            }
            label {
                font-weight: bold;
                margin-bottom: 5px;
            }
            input[type="number"] {
                padding: 10px;
                font-size: 1em;
                border-radius: 8px;
                border: 1px solid #ccc;
                width: 100%;
                box-sizing: border-box;
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
                margin-top: 20px;
            }
            input[type="submit"]:hover {
                background-color: #45a049;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🏡 House Price Prediction API</h2>
            <div class="subtitle">💚 Predict your dream home's worth in seconds!</div>
            <form action="/predict" method="post">
                <div>
                    <label for="bedrooms">🛏 Bedrooms</label>
                    <input type="number" name="bedrooms" min="0" required>
                </div>
                <div>
                    <label for="bathrooms">🛁 Bathrooms</label>
                    <input type="number" step="0.5" name="bathrooms" min="0" required>
                </div>
                <div>
                    <label for="sqft_living">📐 Sqft Living</label>
                    <input type="number" name="sqft_living" min="0" required>
                </div>
                <div>
                    <label for="floors">🏢 Floors</label>
                    <input type="number" name="floors" min="1" required>
                </div>
                <div>
                    <label for="condition">🔍 Condition (1 to 5)</label>
                    <input type="number" name="condition" min="1" max="5" required>
                </div>
                <div>
                    <label for="age">📅 Age of House</label>
                    <input type="number" name="age" min="0" required>
                </div>
                <div>
                    <label for="is_renovated">🔧 Renovated? (0 = No, 1 = Yes)</label>
                    <input type="number" name="is_renovated" min="0" max="1" required>
                </div>
                <input type="submit" value="Predict Price 💰">
            </form>
        </div>
    </body>
    </html>
    ''')


# Single Prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_dict = {
    "bedrooms": int(request.form['bedrooms']),
    "bathrooms": float(request.form['bathrooms']),
    "sqft_living": int(request.form['sqft_living']),
    "floors": int(request.form['floors']),
    "condition": int(request.form['condition']),
    "age": int(request.form['age']),
    "is_renovated": int(request.form['is_renovated'])
}
        features = pd.DataFrame([input_dict])
        prediction = model.predict(features)[0]

        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Prediction Result</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background: #f2f2f2;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                }
                .result-box {
                    background-color: #fff;
                    padding: 40px;
                    border-radius: 12px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    text-align: center;
                }
                .result-box h1 {
                    color: #333;
                }
                .result-box p {
                    font-size: 24px;
                    color: #4CAF50;
                    margin-top: 20px;
                }
            </style>
        </head>
        <body>
            <div class="result-box">
                <h1>Predicted Price</h1>
                <p>${{ prediction }}</p>
                <br><a href="/">Try another</a>
            </div>
        </body>
        </html>
        """
        return render_template_string(html_template, prediction=round(prediction, 2))

    except Exception as e:
        return f"<h2 style='color:red'>Something went wrong: {e}</h2><br><a href='/'>Back</a>"

# Batch Prediction
@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    data = request.json
    df = pd.DataFrame(data)
    df = scaler.transform(df)
    predictions = best_model.predict(df)
    return jsonify({"Predictions": [round(pred, 2) for pred in predictions]})

# ===============================
#        Run Application
# ===============================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
