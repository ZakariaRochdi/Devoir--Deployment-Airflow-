
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
import os

# ------------------- Load Dataset -------------------
dataset = pd.read_csv("./src/hiring.csv")

# ------------------- Clean Data -------------------
dataset["experience"] = dataset["experience"].fillna("zero")
dataset["test_score"] = dataset["test_score"].fillna(dataset["test_score"].mean())

# ------------------- Utils -------------------
def convert_to_int(value):
    if value is None:
        return 0

    if isinstance(value, (int, float)):
        return int(value)

    value = str(value).strip().lower()

    mapping = {
        "zero": 0, "one": 1, "two": 2, "three": 3,
        "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12
    }

    return mapping.get(value, 0)

# ------------------- Feature Engineering -------------------
dataset["experience"] = dataset["experience"].apply(convert_to_int)

# ------------------- Train -------------------
X = dataset[["experience", "test_score", "interview_score"]]
y = dataset["salary"]

model = LinearRegression()
model.fit(X, y)

# ------------------- Save Model -------------------
os.makedirs("./models", exist_ok=True)
joblib.dump(model, "./models/model.joblib")
print("✅ Modèle entraîné et sauvegardé avec succès")
