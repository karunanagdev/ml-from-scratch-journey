import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


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

student = pd.DataFrame({
    "hours": [7],
    "attendance": [90]
})

prediction = model.predict(student)

print("Prediction:", prediction)

probability = model.predict_proba(student)

print("Probability:", probability)

print(model.classes_)
print("Fail probability:", probability[0][0])
print("Pass probability:", probability[0][1])

students = pd.DataFrame({
    "hours": [2, 3, 4, 5, 7, 8],
    "attendance": [50, 60, 65, 70, 85, 90]
})

probabilities = model.predict_proba(students)

print(probabilities)

student = pd.DataFrame({
    "hours": [4],
    "attendance": [70]
})

print(model.predict(student))
print(model.predict_proba(student))