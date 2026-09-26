from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
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


# Small tree
model_small = DecisionTreeClassifier(
    max_depth=2,
    random_state=42
)

model_small.fit(X_train, y_train)

train_pred_small = model_small.predict(X_train)
test_pred_small = model_small.predict(X_test)

print("Small tree")
print("Train accuracy:", accuracy_score(y_train, train_pred_small))
print("Test accuracy:", accuracy_score(y_test, test_pred_small))


# Large tree
model_large = DecisionTreeClassifier(
    max_depth=None,
    random_state=42
)

model_large.fit(X_train, y_train)

train_pred_large = model_large.predict(X_train)
test_pred_large = model_large.predict(X_test)

print("\nLarge tree")
print("Train accuracy:", accuracy_score(y_train, train_pred_large))
print("Test accuracy:", accuracy_score(y_test, test_pred_large))

print("\nSmall tree")
print("Train accuracy:", accuracy_score(y_train, train_pred_small))
print("Test accuracy:", accuracy_score(y_test, test_pred_small))
print("Actual depth:", model_small.get_depth())
print("Leaves:", model_small.get_n_leaves())


print("\nLarge tree")
print("Train accuracy:", accuracy_score(y_train, train_pred_large))
print("Test accuracy:", accuracy_score(y_test, test_pred_large))
print("Actual depth:", model_large.get_depth())
print("Leaves:", model_large.get_n_leaves())

plt.figure(figsize=(12, 8))

plot_tree(
    model_small,
    feature_names=X.columns,
    class_names=model_small.classes_,
    filled=True
)

plt.show()

plt.figure(figsize=(12, 8))

plot_tree(
    model_large,
    feature_names=X.columns,
    class_names=model_large.classes_,
    filled=True
)

plt.show()