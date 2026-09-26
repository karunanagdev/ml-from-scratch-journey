import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


""" data = {
    "hours": [1, 2, 2, 3, 4, 5, 6, 7, 8, 9],
    "attendance": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    "result": [
        "Fail", "Fail", "Fail", "Pass", "Pass",
        "Pass", "Pass", "Pass", "Pass", "Pass"
    ]
} """

data = {
    "hours": [
        1, 2, 2, 3, 3, 4, 4, 5, 5, 6,
        6, 7, 7, 8, 8, 9, 9, 10,
        2, 4, 6, 8, 3, 5, 7, 9
    ],

    "attendance": [
        50, 55, 80, 60, 90, 65, 85, 70, 95, 55,
        75, 60, 85, 70, 90, 65, 80, 95,
        45, 95, 50, 60, 75, 85, 55, 70
    ],

    "result": [
        "Fail", "Fail", "Pass", "Fail", "Pass",
        "Fail", "Pass", "Fail", "Pass", "Fail",
        "Pass", "Pass", "Pass", "Pass", "Pass",
        "Fail", "Pass", "Pass",
        "Fail", "Pass", "Fail", "Pass",
        "Fail", "Pass", "Fail", "Pass"
    ]
}
df = pd.DataFrame(data)

X = df[["hours", "attendance"]]
y = df["result"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



#model = DecisionTreeClassifier(random_state=42)

model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

print("Maximum allowed depth:", 5)
print("Actual tree depth:", model.get_depth())
print("Number of leaves:", model.get_n_leaves())

y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("Actual:", y_test.values)


accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

student = pd.DataFrame({
    "hours": [4],
    "attendance": [68]
})

prediction = model.predict(student)

print("Prediction:", prediction)



plt.figure(figsize=(12, 8))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=model.classes_,
    filled=True
)

plt.show()