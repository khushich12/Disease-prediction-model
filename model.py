import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load dataset
data = pd.read_csv("dataset.csv")

# Features & Target
X = data.drop("disease", axis=1)
y = data["disease"]

# Model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained successfully!")