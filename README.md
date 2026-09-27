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

# Logistic Regression — Probabilities

## Goal

Understand how Logistic Regression uses probabilities to make classification decisions.

## What I Learned

* `predict()` returns the predicted class.
* `predict_proba()` returns the probability for each class.
* `model.classes_` shows the order of the classes.
* Logistic Regression uses a decision threshold to choose the final class.
* Features such as study hours and attendance affect the predicted probability.

## Example

```text
Student
   ↓
Hours + Attendance
   ↓
Logistic Regression
   ↓
Probabilities
   ↓
Pass / Fail
```

## Experiment

Tested different combinations of study hours and attendance and observed how they changed the probability of passing.


# K-Nearest Neighbors (KNN)

## Goal

Learn how KNN makes predictions using the most similar data points.

## What I Learned

* KNN looks at nearby data points to make predictions.
* `n_neighbors` controls how many neighbors are considered.
* KNN can be used for classification.
* Smaller `k` focuses more on nearby examples.
* Larger `k` considers more examples.
* KNN predictions can be evaluated using accuracy.
* Feature scaling is important for distance-based algorithms like KNN.

## Workflow

```text
Data
 ↓
Train/Test Split
 ↓
KNN
 ↓
Find nearest neighbors
 ↓
Vote
 ↓
Prediction
 ↓
Accuracy
```

## Experiment

Tested different values of `k` such as `1`, `3`, `5`, and `7` and observed how they affected predictions and accuracy.


# Feature Scaling

## Goal

Learn why feature scaling is important for distance-based algorithms like KNN.

## What I Learned

* Features can have very different scales.
* KNN uses distances between data points.
* A feature with a larger numerical range can have more influence on distance.
* `StandardScaler` scales features to a similar range.
* `fit_transform()` is used on training data.
* `transform()` is used on test data.
* Feature scaling is especially important for algorithms based on distance.

## Workflow

Data → Train/Test Split → Scale Features → KNN → Predict → Evaluate

## Experiment

Compared KNN predictions and accuracy with and without feature scaling using different values of `k`.

## Key Idea

Feature scaling helps prevent features with larger numerical values from dominating distance calculations.

# Decision Tree

## Goal

Learn how Decision Trees make classification predictions using a series of questions or rules.

## What I Learned

* A Decision Tree makes predictions by asking a sequence of questions.
* Each split divides the data into smaller groups.
* Leaf nodes contain the final prediction.
* `max_depth` controls the maximum depth of the tree.
* `model.get_depth()` shows the actual depth of the trained tree.
* `model.get_n_leaves()` shows the number of leaf nodes.
* `plot_tree()` can be used to visualize the tree.
* Decision Trees can be used for classification.

## Workflow

Data → Train/Test Split → Decision Tree → Train → Predict → Evaluate

## Experiment

Experimented with different `max_depth` values and observed how the tree structure changes.

## Key Idea

`max_depth` is a **maximum limit**, not a requirement. The tree may stop earlier if additional splits are not useful.

# Overfitting

## Goal

Understand how a machine learning model can perform very well on training data but perform worse on new data.

## What I Learned

* Training accuracy measures performance on data the model has already seen.
* Test accuracy measures performance on unseen data.
* A model can memorize training data instead of learning general patterns.
* This is called overfitting.
* Decision Trees can overfit when they become too complex.
* `max_depth` can be used to control tree complexity.
* Comparing training and test accuracy helps identify overfitting.

## Experiment

Compared a small Decision Tree with a large Decision Tree.

```text
Small Tree → Limited complexity
Large Tree → More complexity
```

The goal was to observe how increasing model complexity can improve training accuracy while potentially reducing test accuracy.

## Key Idea

A model should not simply memorize the training data.

```text
Good model → Learns patterns → Works well on new data
Overfitted model → Memorizes data → Performs worse on new data
```

## Workflow

Data → Train/Test Split → Train Models → Compare Train Accuracy → Compare Test Accuracy

# Random Forest

## Goal

Learn how Random Forest uses multiple Decision Trees to make predictions.

## What I Learned

* Random Forest is an ensemble machine learning algorithm.
* It combines multiple Decision Trees.
* Each tree makes a prediction.
* The trees' predictions are combined to produce the final prediction.
* `n_estimators` controls the number of trees in the forest.
* Random Forest can be used for classification.
* More trees do not always mean higher accuracy.

## Workflow

Data → Train/Test Split → Random Forest → Multiple Decision Trees → Combine Predictions → Evaluate

## Experiment

Tested different values of `n_estimators`, such as 1, 5, 100, and 500, and compared the predictions and accuracy.

## Key Idea

```text
Decision Tree → One tree → Prediction

Random Forest → Many trees → Combined prediction
```
# Naive Bayes

## Goal

Learn how Naive Bayes uses probabilities to make classification predictions.

## What I Learned

* Naive Bayes is a classification algorithm.
* It uses probabilities to predict the most likely class.
* `GaussianNB` can be used for numerical features.
* `model.fit()` trains the model.
* `model.predict()` predicts the class.
* `model.predict_proba()` shows the probability for each class.
* Naive Bayes can be used for problems with multiple features.

## Workflow

Data → Train/Test Split → GaussianNB → Train → Predict → Evaluate

## Experiment

Tested different combinations of study hours and attendance and observed how Naive Bayes predicted Pass or Fail.

## Key Idea

```text
Features
   ↓
Calculate probabilities
   ↓
Compare classes
   ↓
Choose most likely class
   ↓
Prediction
```

# Support Vector Machine (SVM)

## Goal

Learn how Support Vector Machines (SVM) classify data by finding a decision boundary between classes.

## What I Learned

* SVM is a classification algorithm.
* SVM finds a decision boundary that separates different classes.
* SVM tries to create a good margin between classes.
* Support vectors are the data points closest to the decision boundary.
* Feature scaling is important for SVM.
* `StandardScaler` can be used to scale features.
* `SVC()` is used to create an SVM classifier.
* Different kernels can create different decision boundaries.
* `C` controls the trade-off between a wider margin and training errors.
* `gamma` controls how flexible an RBF kernel can be.
* A Pipeline can combine preprocessing and a model.

## Workflow

Data → Train/Test Split → StandardScaler → SVM → Predict → Evaluate

## Important Parameters

### `kernel`

Controls the type of decision boundary.

* `linear` → straight boundary
* `rbf` → flexible, nonlinear boundary

### `C`

Controls how strongly the model tries to avoid training errors.

* Small `C` → more tolerant of errors
* Large `C` → stricter about training errors

### `gamma`

Used mainly with the RBF kernel.

* Small `gamma` → smoother boundary
* Large `gamma` → more complex/local boundary

## Experiment

Tested different SVM configurations:

```python
SVC(kernel="linear")
SVC(kernel="rbf")
SVC(kernel="rbf", C=0.1)
```

Compared their predictions and accuracy.

## Key Idea

Features → Scale Features → SVM → Find Decision Boundary → Prediction

SVM tries to separate classes while maintaining a useful margin between them.

# Cross-Validation

## Goal

Learn how Cross-Validation evaluates a machine learning model using multiple train/test splits.

## What I Learned

* A single train/test split can sometimes give a lucky or unlucky result.
* Cross-Validation evaluates the model multiple times.
* `cv=5` means the data is divided into 5 folds.
* Each fold is used as the test set once.
* The model is trained and evaluated multiple times.
* `cross_val_score()` can calculate the score for each fold.
* `.mean()` can be used to calculate the average score.
* Cross-Validation gives a more reliable estimate of model performance.

## Workflow

Dataset → Split into Folds → Train & Test Multiple Times → Get Scores → Average Score

## Example

```python
scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("Scores:", scores)
print("Average accuracy:", scores.mean())
```

## Experiment

Tested different values of `cv`, such as:

```python
cv=3
cv=5
cv=10
```

and observed how the scores and average accuracy changed.

## Key Idea

Train/Test Split:

```text
One split → One evaluation
```

Cross-Validation:

```text
Multiple splits → Multiple evaluations → Average performance
```

Cross-Validation is especially useful when comparing models and tuning hyperparameters.
