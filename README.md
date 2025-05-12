# 🏡 Advanced House Price Prediction Model with MLOps

## 📂 Project Overview

This project is a Flask-based web application that predicts house prices using machine learning. It offers:

* A web-based frontend for user input
* REST API endpoints for prediction
* A trained ML model using scikit-learn
* A containerized deployment via Docker


## 📚 Project Structure

```
.
├── app.py                 # Main Flask application
├── house_prices.csv       # Dataset used for training
├── models/
│   └── house_price_model.pkl  # Saved trained model
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker container instructions
```

## 📁 Code Explanation (app.py)

### ✉️ Imports

```python
import pandas as pd, numpy as np, os, json, joblib
import seaborn as sns, matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template_string
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from flasgger import Swagger
import shap
```

These libraries handle data processing, ML modeling, visualization, API development, and documentation.

### 📅 Data Management

```python
data = pd.read_csv('house_prices.csv')
data.ffill(inplace=True)
```

Loads the dataset and fills missing values using forward fill.

#### 💡 Feature Engineering

```python
data['age'] = 2025 - data['yr_built']
data['is_renovated'] = data['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)
data['price_per_sqft'] = data['price'] / data['sqft_living']
```

Adds new features: age of the house, renovation status, and price per square foot.

#### ⚖️ Feature & Target Selection

```python
X = data[['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'condition', 'age', 'is_renovated']]
y = data['price']
```

### 🔎 Data Splitting & Scaling

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

Splits data and normalizes features.

### 💡 Model Training

```python
models = {
  "RandomForest": RandomForestRegressor(...),
  "GradientBoosting": GradientBoostingRegressor(...)
}
```

Trains two models and compares them using MSE, MAE, and R2.

```python
best_model_name = min(model_scores, key=lambda x: model_scores[x]["MSE"])
best_model = models[best_model_name]
```

Selects the best performing model and saves it with joblib.


## 🚀 API Development

### Flask App Initialization

```python
app = Flask(__name__)
Swagger(app)
```

Creates a Flask app and initializes Swagger for API docs.

### 🏛 Home Route

Displays a styled HTML form to accept user input:

* Emojis + Form fields for bedrooms, bathrooms, sqft, etc.
* Responsive design using CSS

### 🔢 /predict Route

Processes form submission, converts input into a DataFrame, predicts using the model, and returns the predicted price in a styled result page.

### 🤜 /batch\_predict Route

Takes JSON list of records, predicts for all, and returns a list of prices.


## 🚧 Dockerfile Explanation

```Dockerfile
# Use Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose app port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
```

This Dockerfile sets up a lightweight container with Python 3.10, installs the required packages, exposes port 5000, and launches the Flask app.


## 🚜 Running the App in Docker

1. **Build the image**:

```bash
docker build -t house-price-app .
```

2. **Run the container**:

```bash
docker run -p 5000:5000 house-price-app
```

## 🚀 Future Enhancements

* Add SHAP visualizations to explain predictions
* Integrate CI/CD pipeline
* Deploy to cloud (AWS/GCP/Azure)
* Add database for storing predictions


## 🧱 Team Notes

This documentation explains every major section of the code and container setup. Feel free to update the feature engineering logic or model evaluation metrics as needed. For deployment, ensure the Docker image is pushed to a registry if used in a production pipeline.
