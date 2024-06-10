import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from alibi_detect.cd import KSDrift
from alibi_detect.utils.data import create_outlier_batch
import numpy as np

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define objective function for Optuna
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    }

    with mlflow.start_run():
        # Train a classifier
        clf = RandomForestClassifier(**params)
        clf.fit(X_train, y_train)

        # Log parameters
        mlflow.log_params(params)

        # Evaluate model
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)

        # Log the trained model
        mlflow.sklearn.log_model(clf, "random_forest_model")

        # Detect data drift using Alibi-Detect
        drift_detector = KSDrift(X_train)
        drift_results = drift_detector.predict(X_test)

        # Log data drift results to MLFlow
        mlflow.log_metric("data_drift", np.mean(drift_results['data']['is_drift']))

    return accuracy

# Setup Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

# Print the best parameters found
print("Best parameters:", study.best_params)
