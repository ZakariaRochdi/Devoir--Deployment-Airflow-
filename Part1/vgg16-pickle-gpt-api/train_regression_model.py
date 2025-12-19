import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("./src/hiring.csv")

dataset["experience"].fillna(0, inplace=True)
dataset["test_score"].fillna(dataset["test_score"].mean(), inplace=True)

def convert_to_int(word):
    mapping = {
        "zero": 0, "one": 1, "two": 2, "three": 3,
        "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, 0: 0
    }
    return mapping[word]

dataset["experience"] = dataset["experience"].apply(convert_to_int)

X = dataset.iloc[:, :3]
y = dataset.iloc[:, -1]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "./models/model.joblib")
print("✅ Modèle sauvegardé")
