# MLFlow Integration with Optuna and Alibi-Detect

This repository contains a Python script demonstrating the integration of MLFlow for experiment tracking, Optuna for hyperparameter optimization, and Alibi-Detect for detecting data drift in a machine learning pipeline.

## Overview

In this project, we utilize the Iris dataset for multi-class classification and train a Random Forest classifier. Here's a breakdown of the components:

- **MLFlow**: MLFlow is used for experiment tracking. We log the parameters, metrics, and trained models to MLFlow for each experiment run.
- **Optuna**: Optuna is employed for hyperparameter optimization. We set up an Optuna study to search for the best hyperparameters for the Random Forest classifier.
- **Alibi-Detect**: Alibi-Detect's KSDrift detector is utilized to detect data drift between the training and test datasets. We log the data drift results to MLFlow for monitoring.

## Requirements

Make sure you have the following packages installed in your Python environment:

- mlflow
- optuna
- alibi-detect
- pandas
- scikit-learn
- numpy

You can install these dependencies using pip:

```bash
pip install mlflow optuna alibi-detect pandas scikit-learn numpy
```


This script will train a Random Forest classifier on the Iris dataset, optimize its hyperparameters using Optuna, and detect data drift using Alibi-Detect. The results will be logged to MLFlow for experiment tracking and analysis.

## Results

After running the script, you can view the experiment results in the MLFlow tracking UI by navigating to [http://localhost:5000](http://localhost:5000/). You'll be able to see the logged parameters, metrics, and models for each experiment run.
