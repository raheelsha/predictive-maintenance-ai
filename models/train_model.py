import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os

df=pd.read_csv('data/sensor_data.csv')
features = ["temperature", "vibration", "pressure", "rotation_speed"]

X=df[features]
y=df["failure"]     #Logic: In Machine Learning, X usually represents the "Inputs" (the sensors) and y represents the "Target" (what we want to predict: Failure).

# we use the isolation forest This model is Unsupervised. It doesn't look at the failure label; it just looks for things that look "strange."
iso_forest = IsolationForest(contamination=0.1, random_state=42) 
iso_forest.fit(X)
df['anomaly_score'] = iso_forest.decision_function(X)
df['anomaly_flag'] = (iso_forest.predict(X) == -1).astype(int)

X_train ,X_test ,y_train ,y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

#Predicting on the test set
y_pred = rf_model.predict(X_test)
print("=== Random Forest Results ===")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

os.makedirs('models', exist_ok=True)
with open('models/rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
with open('models/iso_forest.pkl', 'wb') as f:
    pickle.dump(iso_forest, f)

print("\nModels saved successfully.")