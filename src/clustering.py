from sklearn.cluster import KMeans
import pandas as pd


def cluster_customers(rfms):
    """
    Cluster customers into 'Good' and 'Bad' using KMeans.
    """
    # Ensure Recency is numeric
    if not pd.api.types.is_numeric_dtype(rfms['Recency']):
        # Convert to numeric if needed
        rfms['Recency'] = rfms['Recency'].dt.days

    # Ensure all columns are numeric
    numeric_columns = ['Recency', 'Frequency', 'Monetary', 'Stability']
    rfms[numeric_columns] = rfms[numeric_columns].apply(
        pd.to_numeric, errors='coerce')

    # Handle missing values
    rfms.fillna(0, inplace=True)

    # Perform clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    rfms['Cluster'] = kmeans.fit_predict(rfms[numeric_columns])
    rfms['Label'] = rfms['Cluster'].map(
        {0: 'Good', 1: 'Bad'})  # Map cluster to labels

    return rfms
