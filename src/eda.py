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
        """
        Calculate and return descriptive statistics for the dataset.
        This method uses the pandas `describe()` function to generate
        descriptive statistics for the DataFrame stored in `self.data`.
        The statistics include measures such as count, mean, standard 
        deviation, min, max, and percentiles.
        Returns:
            pandas.DataFrame: A DataFrame containing the descriptive 
            statistics of the dataset.
        """
        return self.data.describe()

    def get_correlation(self, *args):
        """
        Calculate and return the correlation matrix for numerical columns in the dataset.
        This method selects numerical columns (int64 and float64) from the dataset and computes
        the correlation matrix for these columns.
        Args:
            *args: Additional arguments (not used in this method).
        Returns:
            pandas.DataFrame: A DataFrame containing the correlation matrix of the numerical columns.
        """
        numericaldata = self.data.select_dtypes(include=['int64', 'float64'])
        correlation = numericaldata.corr()
        return correlation

    def get_missing_values(self):
        """
        Calculate and return the number of missing values in each column of the dataset.
        This method computes the sum of missing values for each column and returns a series
        containing only the columns with missing values.
        Returns:
            pandas.Series: A series indicating the number of missing values in each column.
        """
        missing_values = self.data.isnull().sum()
        missing_values = missing_values[missing_values > 0]
        return missing_values

    def get_categorical_distribution(self):
        """
        Computes the distribution of categorical features in the dataset.
        Returns:
            dict: A dictionary where keys are column names and values are 
                  Series containing the counts of unique values in each 
                  categorical column.
        """
        categoricaldata = self.data.select_dtypes(
            include=['object', 'category'])
        distribution = {}
        for column in categoricaldata.columns:
            distribution[column] = categoricaldata[column].value_counts()
        return distribution

    def delete_columns(self, columns):
        """
        Delete columns from the dataset.
        This method deletes the specified columns from the dataset and returns the modified dataset.
        Args:
            columns (list): A list of column names to be deleted from the dataset.
        Returns:
            pandas.DataFrame: A DataFrame with the specified columns removed.
        """
        self.data.drop(columns, axis=1)
        return self.data


class EdaPlot(EdaAnalysis):
    """
    A class used to perform exploratory data analysis (EDA) and generate various plots.
    Attributes
    ----------
        data : pandas.DataFrame
            The dataset to be analyzed and plotted.
    Methods
    -------
        plot_correlation_matrix(target):
            Plots the correlation matrix of the dataset.
        plot_boxplot(column):
            Plots a boxplot for a specified column in the dataset.
        plot_numerical_distribution():
            Plots the distribution of all numerical columns in the dataset.
        plot_categorical_distribution():
            Plots the distribution of all categorical columns in the dataset.
        """

    def plot_correlation_matrix(self):
        """
        Plots the correlation matrix of the dataset.
        This method creates an instance of the EdaAnalysis class using the provided
        dataset and then plots the correlation matrix using matplotlib's matshow function.
        Returns:
            None
        """
        eda = EdaAnalysis(self.data)
        plt.matshow(eda.get_correlation())
        plt.show()

    def plot_boxplot(self, column):
        """
        Plots a boxplot for the specified column in the dataset.
        Parameters:
        column (str): The name of the column to plot the boxplot for.
        Returns:
        None
        """
        self.data.boxplot(column=column)
        plt.show()

    def plot_numerical_distribution(self, column):
        """
        Plots the distribution of numerical features in the dataset.
        This method selects all numerical columns (int64 and float64) from the dataset
        and creates a histogram for each column to visualize its distribution. Each histogram
        is displayed with a title, x-axis label, and y-axis label.
        Parameters:
        column (str): The name of the column to plot the distribution for.
        Returns:
        None
        """
        plt.figure(figsize=(10, 6))
        self.data[column].hist(bins=30, edgecolor='black')
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

    def plot_categorical_distribution(self, column):
        """
        Plots the distribution of categorical features in the dataset.
        This method selects all columns of type 'object' or 'category' from the dataset
        and plots a bar chart for the distribution of each categorical feature.
        Parameters:
        column (str): The name of the column to plot the distribution for.
        Returns:
        None
        """
        plt.figure(figsize=(10, 6))
        self.data[column].value_counts().plot(kind='bar')
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()
