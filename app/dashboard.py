import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Predictive Maintenance AI", layout="wide")

@st.cache_resource
def load_models():
    with open('models/rf_model.pkl', 'rb') as f:
        rf = pickle.load(f)
    with open('models/iso_forest.pkl', 'rb') as f:
        iso = pickle.load(f)
    return rf, iso

@st.cache_data
def load_data():
    return pd.read_csv('data/sensor_data.csv', parse_dates=['timestamp'])

rf_model, iso_forest = load_models()
df = load_data()

features = ['temperature', 'vibration', 'pressure', 'rotation_speed']
df['predicted_failure'] = rf_model.predict(df[features])
df['anomaly_score'] = iso_forest.decision_function(df[features])
df['health_score'] = np.interp(df['anomaly_score'], 
                                [df['anomaly_score'].min(), df['anomaly_score'].max()], 
                                [0, 100]).round(1)

st.title("🏭 AI Predictive Maintenance Dashboard")
st.markdown("Real-time machine health monitoring powered by Machine Learning")

col1, col2, col3, col4 = st.columns(4)
latest = df.iloc[-1]
col1.metric("Temperature", f"{latest['temperature']}°C")
col2.metric("Vibration", f"{latest['vibration']} g")
col3.metric("Pressure", f"{latest['pressure']} PSI")
col4.metric("Rotation Speed", f"{latest['rotation_speed']} RPM")

st.markdown("---")

health = df['health_score'].iloc[-1]
status = "🔴 CRITICAL" if health < 40 else "🟡 WARNING" if health < 70 else "🟢 HEALTHY"
st.subheader(f"Machine Status: {status}")
st.progress(int(health) / 100)
st.write(f"Health Score: {health}/100")

st.markdown("---")

st.subheader("Sensor Trends")
sensor = st.selectbox("Select Sensor", features)
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['timestamp'], y=df[sensor], mode='lines', name=sensor))
failure_points = df[df['predicted_failure'] == 1]
fig.add_trace(go.Scatter(x=failure_points['timestamp'], y=failure_points[sensor],
                          mode='markers', marker=dict(color='red', size=4), name='Failure Zone'))
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

st.subheader("⚠️ Failure Alerts")
alerts = df[df['predicted_failure'] == 1][['timestamp', 'temperature', 'vibration', 'pressure', 'rotation_speed', 'health_score']]
if len(alerts) > 0:
    st.error(f"🚨 {len(alerts)} failure-risk readings detected in the last 30 days")
    st.dataframe(alerts.tail(10), use_container_width=True)
else:
    st.success("No failures detected")
    