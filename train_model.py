import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
import os

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

# Load your diabetes data
df = pd.read_csv("data/diabetes.csv")

# Split into features and label
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model to a .pkl file
with open("model/diabetes_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved to model/diabetes_model.pkl")
