import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score


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


model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier()
)


param_grid = {
    "kneighborsclassifier__n_neighbors": [1, 3, 5]
}


grid_search = GridSearchCV(
    model,
    param_grid,
    cv=3,
    scoring="accuracy"
)


grid_search.fit(X_train, y_train)


print("Best parameters:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)


best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Test accuracy:", accuracy)