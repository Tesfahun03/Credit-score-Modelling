import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

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
        perform(self, method):
        aggregate(self, column, operation):
    """
    logging.info("Creating the AggregateFeature class")

    def __init__(self, data):
        self.data = data

    # method to aggregate a column based on a group by column and specified operation
    def aggregate(self, group_by, column, operation, new_column_name):
        """
        Aggregates data based on the specified column and operation.
        Parameters:
            column (str): The name of the column to aggregate.
            operation (str): The aggregation operation to perform (e.g., 'sum', 'mean', 'max', etc.).
            group_by (str): The column name to group the data by.
        Returns:
            pandas.Series: The result of the aggregation operation.
        """
        logging.info(
            "Aggregating data based on specified column and operation")
        aggregated = self.data.groupby(group_by)[column].transform(operation)
        column_index = self.data.columns.get_loc(column)
        self.data.insert(column_index + 1, new_column_name, aggregated)


