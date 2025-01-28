import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)

# a class used to aggregate features into a new columns


class AggregateFeature:
    """
    AggregateFeature class for performing aggregation operations on a dataset.
    Attributes:
        data (pandas.DataFrame): The dataset on which aggregation operations will be performed.
    Methods:
        __init__(self, data, group_by):
            Initializes the AggregateFeature class with the provided data and group_by column.
        perform_insertion(self, column, new_column_name, value):performs an insertion operation on the specified column.
        aggregate(self, column, operation):
    """
    logging.info("Creating the AggregateFeature class")

    def __init__(self, data):
        self.data = data

    def perform_insertion(self, column, new_column_name, value):
        """
        Performs an insertion operation on the specified column.
        Parameters:
            new_column_name (str): The name of the new column to insert.
            value (any): The value to insert into the new column.
            column (str): The name of the column to insert.
        """
        logging.info(
            f"Performing insertion operation on {new_column_name} column")
        column_index = self.data.columns.get_loc(column)
        self.data.insert(column_index + 1, new_column_name, value)

    # method to aggregate a column based on a group by column and specified operation

    def aggregate(self, group_by, column, operation, new_column_name):
        """
        Aggregates data based on the specified column and operation.
        Parameters:
            group_by (str): The name of the column to group by.
            column (str): The name of the column to aggregate.
            new_column_name (str): The name of the new column to store the aggregated results.
        Returns:
            pandas.Series: The result of the aggregation operation.
        """
        logging.info(
            f"Aggregating data  on {column} and operation: {operation}")
        aggregated = self.data.groupby(group_by)[column].transform(operation)
        column_index = self.data.columns.get_loc(column)
        self.perform_insertion(column, new_column_name, aggregated)


class ExtractFeature(AggregateFeature):
    logging.info("Creating the ExtractFeature class and initialize data")

    def extract_hr_to_year(self, column, new_column):
        """
        Extracts the hour from a datetime column.
        Returns:
            pandas.Series: A series containing the extracted hour values.
        """
        logging.info("Extracting hour from datetime column")

        extracted = {}
        extracted[1] = self.data[column].dt.hour
        extracted[2] = self.data[column].dt.day
        extracted[3] = self.data[column].dt.month
        extracted[4] = self.data[column].dt.year
        for i, column_names in enumerate(new_column, start=1):
            self.perform_insertion(column, column_names, extracted[i])

        return self.data


class RFMSFeature:
    def __init__(self, data):
        self.data = data

    def calculate_rfms(data):
        """
        Calculate Recency, Frequency, Monetary, and Stability features for each customer.
        """
        current_time = datetime.now()

        # Recency: Days since the last transaction
        recency = data.groupby('CustomerId')['Transaction-Day'].max().apply(lambda x: (current_time - x))

        # Frequency: Number of transactions
        frequency = data.groupby('CustomerId')['TransactionId'].count()

        # Monetary: Total transaction value
        monetary = data.groupby('CustomerId')['Value'].sum()

        # Stability: Standard deviation of transaction values
        stability = data.groupby('CustomerId')['Value'].std().fillna(0)

        # Combine features into a DataFrame
        rfms = pd.DataFrame({
            'Recency': recency,
            'Frequency': frequency,
            'Monetary': monetary,
            'Stability': stability
        }).reset_index()

        return rfms