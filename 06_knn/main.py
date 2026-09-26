import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

data = {
    "hours": [1, 2, 2, 3, 4, 5, 6, 7, 8, 9],
    "attendance": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    "result": [
        "Fail",
        "Fail",
        "Fail",
        "Pass",
        "Pass",
        "Pass",
        "Pass",
        "Pass",
        "Pass",
        "Pass"
    ]
}

df = pd.DataFrame(data)

print(df)

X = df[["hours", "attendance"]]
y = df["result"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_train, y_train)

student = pd.DataFrame({
    "hours": [4],
    "attendance": [68]
})

prediction = model.predict(student)

print("Prediction:", prediction)
print(model.predict_proba(student))

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)