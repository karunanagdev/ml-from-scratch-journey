import pandas as pd
from sklearn.linear_model import LinearRegression


# Create dataset
data = {
    "hours": [1, 2, 3, 4, 5, 6],
    "score": [35, 42, 50, 58, 65, 72]
}

df = pd.DataFrame(data)

print(df)


# Separate features and target
X = df[["hours"]]
y = df["score"]


# Create model
model = LinearRegression()


# Train model
model.fit(X, y)


# Make prediction
prediction = model.predict(pd.DataFrame({"hours": [20]}))

print("Predicted score:", prediction[0])