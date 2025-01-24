import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class EdaAnalysis:
    """
    A class used to perform exploratory data analysis (EDA) on a given dataset.
    Attributes
    ----------
        data : pandas.DataFrame
            The dataset on which EDA is to be performed.
    Methods:
    -------
        get_descriptive_statistics():
            Returns descriptive statistics of the dataset.
        get_correlation(*args):
            Returns the correlation matrix of numerical features in the dataset.
        get_missing_values():
            Returns a series indicating the number of missing values in each column.
        get_categorical_distribution():
            Returns a dictionary with the distribution of categorical features.
    """

    def __init__(self, data):
        self.data = data

    def get_descriptive_statistics(self):
        return self.data.describe()

    def get_correlation(self, *args):
        numerical_data = self.data.select_dtypes(include=['int64', 'float64'])
        correlation = numerical_data.corr()

        return correlation

    def get_missing_values(self):
        missing_values = self.data.isnull().sum()
        missing_values = missing_values[missing_values > 0]
        return missing_values

    def get_categorical_distribution(self):
        categorical_data = self.data.select_dtypes(
            include=['object', 'category'])
        distribution = {}
        for column in categorical_data.columns:
            distribution[column] = categorical_data[column].value_counts()
        return distribution


class EdaPlot:
    def __init__(self, data):
        self.data = data

    def get_data_info(self):
        return self.data.info()

    def plot_missing_values(self):
        missing_values = self.data.isnull().sum()
        missing_values = missing_values[missing_values > 0]
        missing_values.sort_values(inplace=True)
        missing_values.plot.bar()
        plt.show()

    def plot_correlation_matrix(self, target):
        eda = EdaAnalysis(self.data)
        plt.matshow(eda.get_correlation())

    def plot_boxplot(self, column):
        self.data.boxplot(column)
        plt.show()

    def plot_numerical_distribution(self):
        numerical_data = self.data.select_dtypes(include=['int64', 'float64'])
        for column in numerical_data.columns:
            plt.figure(figsize=(10, 6))
            plt.hist(numerical_data[column], bins=30, edgecolor='k', alpha=0.7)
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.show()
