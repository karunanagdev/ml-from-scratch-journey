import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


data = {
    "hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "score": [35, 42, 50, 58, 65, 72, 78, 85, 90, 95]
}

df = pd.DataFrame(data)

print(df)

X = df[["hours"]]
y = df["score"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data:")
print(X_train)

print("\nTesting data:")
print(X_test)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Actual:", y_test.values)
print("Predicted:", y_pred)

mae = mean_absolute_error(y_test, y_pred)

print("MAE:", mae)