import pandas as pd
import random
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import DataSynthetizer as DS

df = DS.synthetizer()

X = df[['Forward_Active_Energy']]
y = df['Reverse_Active_Energy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = IsolationForest()
model.fit(X_train, y_train)

preds = model.predict(X_test)

selected_columns = ["Forward_Active_Energy", "Reverse_Active_Energy"]

X = df[selected_columns]

isolation_forest = IsolationForest(contamination=0.054)
isolation_forest.fit(X)
anomaly_scores = isolation_forest.predict(X)

df["Anomaly_Score"] = anomaly_scores

anomalies = df[df["Anomaly_Score"] == -1]

