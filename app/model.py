import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

def train_and_save_model():
    df = pd.read_csv("data/diabetes.csv")
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    model = RandomForestClassifier(
    n_estimators=100,        # Number of trees in the forest
    max_depth=10,            # Maximum depth of the tree
    min_samples_split=5,     # Minimum number of samples required to split an internal node
    min_samples_leaf=2,      # Minimum number of samples required to be at a leaf node
    max_features='sqrt',     # Number of features to consider when looking for the best split
    bootstrap=True,          # Whether bootstrap samples are used when building trees
    random_state=42          # Seed for reproducibility
    )

    model.fit(X, y)

    with open("model/diabetes_model.pkl", "wb") as f:
        pickle.dump(model, f)

def load_model():
    with open("model/diabetes_model.pkl", "rb") as f:
        return pickle.load(f)
