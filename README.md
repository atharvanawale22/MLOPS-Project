# House Price Prediction Model

This project is a **House Price Prediction Model** powered by **Machine Learning** and integrated with **MLOps components** like Docker, Prometheus, and Grafana for monitoring and container orchestration.

## 📌 **Project Overview**

The model predicts house prices based on various features like the number of bedrooms, bathrooms, square footage, floors, and more. It is built using Flask as a web application, Docker for containerization, Prometheus for monitoring, and Grafana for visualizing metrics.

---

## 🛠 **Technology Stack**

* **Flask** - Web application framework
* **Pandas & NumPy** - Data manipulation and calculations
* **Scikit-Learn** - Machine Learning models
* **Docker** - Containerization
* **Prometheus** - Metrics collection and monitoring
* **Grafana** - Metrics visualization

---

## ⚙️ **MLOps Components**

### Docker

Docker is used to containerize the application, ensuring it runs consistently across different environments.

### Prometheus

Prometheus is set up to scrape metrics from the Flask application. Metrics collected include:

* **prediction\_count**: Number of predictions made
* **prediction\_latency\_seconds**: Latency for predictions in seconds

### Grafana

Grafana is configured to visualize metrics collected by Prometheus, allowing real-time monitoring of the model's performance.

---

## 🚀 **Setup Instructions**

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd House-Price-Prediction
   ```

2. Build and run the Docker containers:

   ```bash
   docker-compose up --build
   ```

3. Access the following services:

   * Flask App: [http://localhost:5000](http://localhost:5000)
   * Prometheus: [http://localhost:9090](http://localhost:9090)
   * Grafana: [http://localhost:3000](http://localhost:3000)

4. Grafana default credentials:

   * **Username**: admin
   * **Password**: admin

---

## 🏷️ **Docker Compose Configuration**

Services configured in `docker-compose.yml`:

* **flask-app**: Hosts the ML model
* **prometheus**: Scrapes metrics from the Flask app
* **grafana**: Visualizes Prometheus metrics

---

## 🔍 **Code Explanation**

### app.py

The main application is built using Flask, with the following components:

1. **Data Loading and Feature Engineering**

   * Data is loaded from `house_prices.csv`.
   * Feature engineering adds new columns like `age`, `is_renovated`, and `price_per_sqft`.

2. **Model Training**

   * Two models are trained: `RandomForestRegressor` and `GradientBoostingRegressor`.
   * The best model is selected based on the lowest Mean Squared Error (MSE) and saved for predictions.

3. **Flask API Setup**

   * `/` → Renders the main prediction form.
   * `/predict` → Accepts a JSON payload and returns a price prediction.
   * `/batch_predict` → Accepts multiple JSON objects for batch predictions.

4. **Prometheus Metrics**

   * Integrated with Prometheus to monitor:

     * `prediction_count`: Tracks the number of predictions.
     * `prediction_latency_seconds`: Measures latency for each prediction.

5. **Middleware Mounting**

   * The Flask app exposes `/metrics` for Prometheus to scrape data every 15 seconds.

---

### docker-compose.yaml

Defines three services:

* **flask-app** → Runs the Flask server.
* **prometheus** → Scrapes metrics from the Flask app.
* **grafana** → Visualizes the metrics collected by Prometheus.

---

### prometheus.yml

Prometheus is configured to scrape data from the Flask app running at `flask-app:5000`. The scrape interval is set to **15 seconds**.

---

## 📊 **Prometheus Monitoring**

Prometheus is configured to scrape metrics from the Flask app every **15 seconds** as specified in `prometheus.yml`. It tracks prediction count and latency metrics which can be queried and visualized in Grafana.

---

## 🌐 **Endpoints**

1. **Home Page**: `/`

   * Displays the form for house price prediction.

2. **Prediction**: `/predict`

   * Accepts JSON payload for prediction.

   * Example:

     ```json
     {
       "bedrooms": 3,
       "bathrooms": 2,
       "sqft_living": 1500,
       "floors": 1,
       "condition": 4,
       "age": 10,
       "is_renovated": 0
     }
     ```

   * Response:

     ```json
     {
       "Predicted Price": 450000.0
     }
     ```

3. **Batch Prediction**: `/batch_predict`

   * Accepts a list of JSON payloads for multiple predictions.
   * Response:

     ```json
     {
       "Predictions": [450000.0, 350000.0, 420000.0]
     }
     ```

4. **Metrics**: `/metrics`

   * Exposes Prometheus metrics for monitoring.

---

## 📌 **Future Enhancements**

* Add more ML models for comparison
* Enable batch processing of predictions
* Deploy on cloud platforms (AWS/GCP)
* Extend monitoring to include memory and CPU usage

---

Feel free to contribute to the project or suggest improvements! 😊

---

## 📬 **Contact**

For any issues, feel free to reach out to the project maintainer.
