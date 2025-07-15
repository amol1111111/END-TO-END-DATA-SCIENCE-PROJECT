import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

def train_and_save_model():
    df = pd.read_csv("data/diabetes.csv")
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    model = RandomForestClassifier()
    model.fit(X, y)

    with open("model/diabetes_model.pkl", "wb") as f:
        pickle.dump(model, f)

def load_model():
    with open("model/diabetes_model.pkl", "rb") as f:
        return pickle.load(f)