import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


data = {
    "hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "score": [35, 42, 50, 58, 65, 72, 78, 85, 90, 95]
}

df = pd.DataFrame(data)

X = df[["hours"]]
y = df["score"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create and train model
model = LinearRegression()

model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)


# Evaluate
mae = mean_absolute_error(y_test, y_pred)

print("Actual:", y_test.values)
print("Predicted:", y_pred)
print("MAE:", mae)

mse = mean_squared_error(y_test, y_pred)

print("MSE:", mse)

rmse = np.sqrt(mse)

print("RMSE:", rmse)

bad_predictions = np.array([20, 50])
bad_mae = mean_absolute_error(y_test, bad_predictions)
bad_mse = mean_squared_error(y_test, bad_predictions)
bad_rmse = np.sqrt(bad_mse)

print("\nBad predictions:")
print("MAE:", bad_mae)
print("MSE:", bad_mse)
print("RMSE:", bad_rmse)