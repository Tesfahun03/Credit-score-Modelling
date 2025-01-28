import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class PlotMetrics:
    """_Plot the accuracy metrics for each model_
    """

    def __init__(self, models, mae_scores, mse_scores, r2_scores, roc_auc_scores):
        # Ensure that models is a list of strings
        self.models = [str(model)
                       for model in models]  # Ensure models are strings
        self.mae_scores = mae_scores
        self.mse_scores = mse_scores
        self.r2_scores = r2_scores
        self.roc_auc_scores = roc_auc_scores  # Add roc_auc_scores as an argument

    def plot(self):
        """_Plot the accuracy metrics for mean absolute error, mean squared error, r2-score, and roc_auc_score_
        """
        logging.info('Plotting graph for accuracy metrics')

        # Plot for Mean Absolute Error (MAE)
        plt.figure(figsize=(6, 4))
        plt.bar(self.models, self.mae_scores, color='green')
        plt.xlabel('Models')
        plt.ylabel('Mean Absolute Error')
        plt.title('Comparison of MAE Scores')
        plt.xticks(rotation=45)
        plt.show()

        # Plot for Mean Squared Error (MSE)
        logging.info('Plotting graph for Mean Squared Error')
        plt.figure(figsize=(6, 4))
        plt.bar(self.models, self.mse_scores, color='yellow')
        plt.xlabel('Models')
        plt.ylabel('Mean Squared Error')
        plt.title('Comparison of MSE Scores')
        plt.xticks(rotation=45)
        plt.show()

        # Plot for R2 Scores
        logging.info('Plotting graph for R2 Scores')
        plt.figure(figsize=(6, 4))
        plt.bar(self.models, self.r2_scores, color='red')
        plt.xlabel('Models')
        plt.ylabel('R2 Scores')
        plt.title('Comparison of R2 Scores')
        plt.xticks(rotation=45)
        plt.show()

        # Check if the roc_auc_scores are numerical values (like float64)
        logging.info(
            f"Checking data type of roc_auc_scores: {type(self.roc_auc_scores)}")

        # Plot for ROC AUC Scores
        logging.info('Plotting graph for ROC AUC Scores')
        if isinstance(self.roc_auc_scores, (list, np.ndarray)):
            plt.figure(figsize=(6, 4))
            plt.bar(self.models, self.roc_auc_scores, color='blue')
            plt.xlabel('Models')
            plt.ylabel('ROC AUC Scores')
            plt.title('Comparison of ROC AUC Scores')
            plt.xticks(rotation=45)
            plt.show()
        else:
            logging.error(
                'roc_auc_scores should be a list or numpy array of numerical values.')
