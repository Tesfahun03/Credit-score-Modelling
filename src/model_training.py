import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class SplitData:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    logging.info('spliting the dataset into 80 and 20% ')

    def split_data(self):
        """_splits the data into 80% of training data and 20% of testing data_

        Returns:
            _train and test data_: _training and testing datas_
        """
        return train_test_split(self.x, self.y, test_size=0.2, random_state=42)


class TrainData:
    """A class for training multiple machine learning algorithms on the same training and testing sets."""

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def random_forest(self):
        """Initialize the Random Forest model and fit the training set."""
        logging.info('Training begins with the training dataset')

        weight = compute_sample_weight(class_weight='balanced', y=self.y_train)
        random_forest_model = RandomForestRegressor(
            n_estimators=100, n_jobs=-1)  # Use all CPU cores
        random_forest_model.fit(
            self.x_train, self.y_train, sample_weight=weight)
        logging.info('Training random forest regressor ends')
        return random_forest_model

    def logistic_regression(self):
        """Initialize the logistic regression model and fit the training set."""
        logging.info('Training logistic regression begins')

        # Ensure target is binary (for classification)
        if self.y_train.dtype.kind in 'fc':  # Check if y_train is continuous
            logging.info(
                'Target variable is continuous, applying thresholding to convert to binary labels')
            threshold = 0.5  # Example threshold for classification
            self.y_train = (self.y_train >= threshold).astype(
                int)  # Convert to binary classes

        # Compute sample weights
        weight = compute_sample_weight(class_weight='balanced', y=self.y_train)

        # Initialize Logistic Regression model
        logistic_model = LogisticRegression(
            n_jobs=-1, class_weight='balanced', random_state=42)  # Use all CPU cores
        logistic_model.fit(self.x_train, self.y_train, sample_weight=weight)

        logging.info('Training logistic regression ends')
        return logistic_model


class EvaluateModel:
    """A class for evaluating machine learning models."""

    def evaluate_model(self, model, x_test, y_test):
        # Make predictions
        y_pred = model.predict(x_test)

        # For regression models, skip roc_auc_score and focus on MAE, MSE, and R2
        if isinstance(model, RandomForestRegressor):
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            roc_auc_scores = None  # Not applicable for regression
        else:
            # For classification models, ensure y_test and y_pred are binary
            if y_test.dtype.kind in 'fc':  # Continuous target
                y_test = (y_test >= 0.5).astype(int)
                y_pred = (y_pred >= 0.5).astype(int)

            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            try:
                roc_auc_scores = roc_auc_score(y_test, y_pred)
            except ValueError as e:
                roc_auc_scores = None  # Handle if roc_auc_score cannot be computed

        return mae, mse, r2, roc_auc_scores, y_pred
