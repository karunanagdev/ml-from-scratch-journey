import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

data = {
    "hours": [1, 2, 2, 3, 4, 5, 6, 7, 8, 9],
    "attendance": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    "result": [
        "Fail", "Fail", "Fail", "Pass", "Pass",
        "Pass", "Pass", "Pass", "Pass", "Pass"
    ]
}

df = pd.DataFrame(data)

X = df[["hours", "attendance"]]
y = df["result"]

model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("Scores:", scores)
print("Average accuracy:", scores.mean())