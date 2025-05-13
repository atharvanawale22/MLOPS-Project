# ===============================
#      House Price Prediction
# ===============================
# 🚀 MLOps Integrated with Docker, Prometheus, and Grafana

# ===============================
#          Import Libraries
# ===============================
import pandas as pd
import numpy as np
import os
import joblib
from flask import Flask, request, jsonify, render_template_string
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from prometheus_client import Counter, Histogram, make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware

# ===============================
#        Prometheus Metrics
# ===============================
prediction_count = Counter('prediction_count', 'Number of Predictions Made')
prediction_latency = Histogram('prediction_latency_seconds', 'Latency for predictions in seconds')

# ===============================
#        Data Management
# ===============================
# Load Dataset
data = pd.read_csv('house_prices.csv')
data.ffill(inplace=True)

# Feature Engineering
data['age'] = 2025 - data['yr_built']
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
models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

model_scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_scores[name] = {
        "MSE": mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2_Score": r2_score(y_test, y_pred)
    }

best_model_name = min(model_scores, key=lambda x: model_scores[x]["MSE"])
best_model = models[best_model_name]

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/house_price_model.pkl')

# ===============================
#        Flask API Setup
# ===============================
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>House Price Prediction</title>
        <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@500&display=swap" rel="stylesheet">
        <style>
            body { font-family: 'Montserrat', sans-serif; background: linear-gradient(135deg, #74ebd5, #9face6); margin: 0; padding: 0; color: #333; }
            .container { max-width: 800px; margin: 60px auto; background: white; border-radius: 15px; box-shadow: 0 0 20px rgba(0,0,0,0.2); padding: 40px; }
            h2 { color: #2c3e50; text-align: center; margin-bottom: 10px; }
            .subtitle { text-align: center; font-size: 1.1em; margin-bottom: 25px; color: #555; }
            form { display: flex; flex-direction: column; gap: 15px; margin-top: 20px; }
            label { font-weight: bold; margin-bottom: 5px; }
            input[type="number"] { padding: 10px; font-size: 1em; border-radius: 8px; border: 1px solid #ccc; width: 100%; box-sizing: border-box; }
            input[type="submit"] { background-color: #4CAF50; color: white; border: none; padding: 12px; font-size: 1em; border-radius: 8px; cursor: pointer; transition: background-color 0.3s; margin-top: 20px; }
            input[type="submit"]:hover { background-color: #45a049; }
            #result { margin-top: 20px; font-size: 1.3em; text-align: center; color: #2c3e50; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🏡 House Price Prediction</h2>
            <div class="subtitle">💚 Predict your dream home's worth in seconds!</div>
            <form id="predictionForm">
                <label>🛏 Bedrooms</label>
                <input type="number" id="bedrooms" required min="0">
                <label>🛁 Bathrooms</label>
                <input type="number" id="bathrooms" step="0.5" required min="0">
                <label>📐 Sqft Living</label>
                <input type="number" id="sqft_living" required min="0">
                <label>🏢 Floors</label>
                <input type="number" id="floors" required min="1">
                <label>🔍 Condition (1-5)</label>
                <input type="number" id="condition" required min="1" max="5">
                <label>📅 Age of House</label>
                <input type="number" id="age" required min="0">
                <label>🔧 Renovated? (0 = No, 1 = Yes)</label>
                <input type="number" id="is_renovated" required min="0" max="1">
                <input type="submit" value="Predict Price 💰">
            </form>
            <div id="result"></div>
        </div>
        <script>
            document.getElementById("predictionForm").addEventListener("submit", async function(event) {
                event.preventDefault();
                const data = {
                    bedrooms: parseInt(document.getElementById("bedrooms").value),
                    bathrooms: parseFloat(document.getElementById("bathrooms").value),
                    sqft_living: parseInt(document.getElementById("sqft_living").value),
                    floors: parseInt(document.getElementById("floors").value),
                    condition: parseInt(document.getElementById("condition").value),
                    age: parseInt(document.getElementById("age").value),
                    is_renovated: parseInt(document.getElementById("is_renovated").value)
                };
                const response = await fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(data)
                });
                const result = await response.json();
                const resultDiv = document.getElementById("result");
                if (response.ok) {
                    resultDiv.innerHTML = `<strong>Predicted Price: $${result["Predicted Price"]}</strong>`;
                } else {
                    resultDiv.innerHTML = `<span style="color:red;">Error: ${result.error}</span>`;
                }
            });
        </script>
    </body>
    </html>
    ''')

@app.route('/predict', methods=['POST'])
@prediction_latency.time()
def predict():
    prediction_count.inc()
    try:
        data = request.json
        features = pd.DataFrame([data])
        features = scaler.transform(features)
        prediction = best_model.predict(features)[0]
        return jsonify({"Predicted Price": round(prediction, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/batch_predict', methods=['POST'])
@prediction_latency.time()
def batch_predict():
    prediction_count.inc()
    try:
        data = request.json
        df = pd.DataFrame(data)
        df = scaler.transform(df)
        predictions = best_model.predict(df)
        return jsonify({"Predictions": [round(pred, 2) for pred in predictions]})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ===============================
#    Prometheus Middleware Mount
# ===============================
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

# ===============================
#        Run Flask Server
# ===============================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
