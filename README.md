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
