# 🏭 AI Predictive Maintenance System

An AI-powered system that predicts industrial machine failures **before they happen** using sensor data analysis and Machine Learning.

## 🚨 Problem
factories lose billions every year due to unexpected machine breakdowns. Current approach is reactive — fix after failure. This system makes it **predictive**.

## 💡 Solution
Analyzes 4 sensor streams (temperature, vibration, pressure, rotation speed) in real-time and predicts failure 24-72 hours in advance.

## 🤖 ML Models
- **Isolation Forest** — unsupervised anomaly detection on sensor patterns
- **Random Forest Classifier** — supervised failure prediction
- **Accuracy: 99%** on test data

## 📊 Dashboard Features
- Live sensor readings
- Machine health score (0-100)
- Failure zone visualization
- Real-time alerts

## 🛠️ Tech Stack
Python, Scikit-learn, Streamlit, Plotly, Pandas, NumPy

## 🚀 Run Locally
```bash
pip install -r requirements.txt
python Data/generate_sensor_data.py
python models/train_model.py
streamlit run app/dashboard.py
```

## 📓 Notebook
Full exploration and model training walkthrough available in `notebooks/`
