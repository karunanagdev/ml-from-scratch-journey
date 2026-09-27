import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


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


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

actual = [
    "Fail",
    "Fail",
    "Fail",
    "Pass",
    "Pass",
    "Pass"
]

predicted = [
    "Fail",
    "Pass",
    "Fail",
    "Pass",
    "Fail",
    "Pass"
]

cm = confusion_matrix(actual, predicted)

print(cm)


from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(
    actual,
    predicted
)

import matplotlib.pyplot as plt

plt.show()