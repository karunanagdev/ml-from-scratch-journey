import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression


# Create dataset
data = {
    "hours": [1, 2, 3, 4, 5, 6],
    "score": [20, 20, 80, 30, 90, 40]
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
y_pred = model.predict(X)

# Make prediction
prediction = model.predict(pd.DataFrame({"hours": [20]}))

print("Predicted score:", prediction[0])

plt.scatter(X, y)

#The dots are the real data.
#The line is what the model learned.
plt.plot(X, y_pred)

plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Linear Regression")

plt.show()

#score = Coefficient × hours + Intercept
print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)