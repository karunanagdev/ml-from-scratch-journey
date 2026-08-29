# ml-from-scratch-journey
Learning traditional machine learning step by step with Python, using small practical examples, experiments, and projects to build a strong foundation.

## What I learned

- Linear Regression learns a relationship between features and a target.
- `X` represents the input/features.
- `y` represents the target.
- `model.fit(X, y)` trains the model.
- `model.predict()` makes predictions.
- Linear Regression learns a straight-line relationship.
- `model.coef_` gives the coefficient.
- `model.intercept_` gives the intercept.
- I can visualize the real data and the learned regression line.


# Train/Test Split

## Goal

Learn how to split data into training and testing sets and evaluate a machine learning model.

## What I Learned

* `X` = features/input
* `y` = target/output
* `train_test_split()` separates data into train and test sets.
* `model.fit()` trains the model using training data.
* `model.predict()` makes predictions on new data.
* MAE measures how far predictions are from actual values.
* `random_state` makes the split reproducible.

## Workflow

```text
Data
 ↓
Train/Test Split
 ↓
Train Model
 ↓
Predict Test Data
 ↓
Evaluate with MAE
```

## Experiment

I experimented with different `test_size` and `random_state` values to see how they affect the results.


# Model Evaluation

## Goal

Learn how to measure how well a regression model performs.

## Metrics

* **MAE** — Average absolute error.
* **MSE** — Squares errors, so large errors are penalized more.
* **RMSE** — Square root of MSE, giving the error in the original units.

## What I Learned

```text
Model
 ↓
Predictions
 ↓
Compare with actual values
 ↓
MAE / MSE / RMSE
```

Lower error generally means better predictions.

## Experiment

Compared the model's predictions with intentionally bad predictions and observed how the evaluation metrics changed.


# Logistic Regression

## Goal

Learn classification using Logistic Regression.

In this example, the model predicts whether a student will **Pass or Fail** based on study hours and attendance.

## What I Learned

* Classification predicts categories instead of numbers.
* `X` contains the features.
* `y` contains the target.
* Logistic Regression can be used for classification.
* `model.fit()` trains the model.
* `model.predict()` predicts a class.
* Accuracy measures how many predictions are correct.

## Features

```text
hours
attendance
```

## Target

```text
Pass / Fail
```

## Workflow

```text
Data
 ↓
X + y
 ↓
Train/Test Split
 ↓
Logistic Regression
 ↓
Train
 ↓
Predict
 ↓
Accuracy
```

## Experiment

Tested different combinations of study hours and attendance to see whether the model predicts **Pass** or **Fail**.
