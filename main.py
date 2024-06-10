import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import *


# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Predict on test set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Log parameters and metrics with MLFlow
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model", "RandomForestClassifier")
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    
    # Log the trained model
    mlflow.sklearn.log_model(clf, "random_forest_model")
    
    # Log dataset features
    mlflow.log_param("feature_names", feature_names)

# Convert the datasets to pandas DataFrame
train_df = pd.DataFrame(X_train, columns=feature_names)
test_df = pd.DataFrame(X_test, columns=feature_names)

# Create data drift report using Evidently AI
report = Report(metrics=[
    DataDriftPreset(), 
])

report.run(reference_data=train_df, current_data=test_df)
report

# Save the report
#drift_report.save('drift_report.html')

# You can also view the report in a notebook
# drift_report.show()
