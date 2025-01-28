import pandas as pd
import numpy as np

def calculate_woe(df, feature, target):
    """
    Calculate WoE for a given feature.
    """
    bins = pd.qcut(df[feature], q=10, duplicates='drop')  # Quantile-based binning
    bin_data = pd.DataFrame({
        'Bin': bins,
        'Good': df[target].eq('Good').groupby(bins).sum(),
        'Bad': df[target].eq('Bad').groupby(bins).sum()
    })
    bin_data['Total'] = bin_data['Good'] + bin_data['Bad']
    bin_data['Good_Pct'] = bin_data['Good'] / bin_data['Good'].sum()
    bin_data['Bad_Pct'] = bin_data['Bad'] / bin_data['Bad'].sum()
    bin_data['WoE'] = np.log(bin_data['Good_Pct'] / bin_data['Bad_Pct'])
    return bin_data

def apply_woe_binning(rfms, target, features):
    """
    Apply WoE binning to all RFMS features.
    """
    woe_results = {}
    for feature in features:
        woe_results[feature] = calculate_woe(rfms, feature, target)
    return woe_results

