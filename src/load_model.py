import os
import sys
import joblib
import numpy as np
sys.path.append(os.path.abspath('..'))
model1 = '../models/logistic_regr.pkl'
model2 = '../models/random_forest.pkl'
# loding our model
xg_model = joblib.load(model1)
rf_model = joblib.load(model2)
