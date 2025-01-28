from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from load_model import xg_model, rf_model
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath('..'))


class UserInput(BaseModel):
    TransactionId: int
    CustomerId: int
    ProviderId: int
    ProductId: int
    ProductCategory: str
    ChannelId: int
    Amount: float
    Value: float
    Standard_Deviation_of_Transaction_Amounts: float
    Transaction_Count: int
    Average_Transaction_Amount: float
    Total_Transaction_Amount: float
    Transaction_Year: int
    Transaction_Month: int
    Transaction_Day: int
    Transaction_Hour: int
    PricingStrategy: str
    FraudResult: bool


app = FastAPI()


# predicting sales using fast API library
@app.post("/predict")
def predict(input: UserInput):
    input_data = np.array([[
        input.TransactionId, input.CustomerId, input.ProviderId, input.ProductId, input.ProductCategory,
        input.ChannelId, input.Amount, input.Value, input.Standard_Deviation_of_Transaction_Amounts,
        input.Transaction_Count, input.Average_Transaction_Amount, input.Total_Transaction_Amount,
        input.Transaction_Year, input.Transaction_Month, input.Transaction_Day, input.Transaction_Hour,
        input.PricingStrategy, input.FraudResult
    ]], dtype=object)

    # Convert to DataFrame for easier processing
    columns = [
        'TransactionId', 'CustomerId', 'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'Amount',
        'Value', 'Standard_Deviation_of_Transaction_Amounts', 'Transaction_Count',
        'Average_Transaction_Amount', 'Total_Transaction_Amount', 'Transaction_Year',
        'Transaction_Month', 'Transaction_Day', 'Transaction_Hour', 'PricingStrategy', 'FraudResult'
    ]
    df = pd.DataFrame(input_data, columns=columns)

    predicted_result = rf_model.predict(df)
    return predicted_result
