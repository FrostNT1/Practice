# Wine Quality Regressor Classifier

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('winequality.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


# Fitting classifier to the Training set
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300)
regressor.fit(X_train, y_train)

# Prediciting the results
ypred = regressor.predict(X_test)

for k in range(len(ypred)):
    if ypred[k] >= 6.5:
        ypred[k] = 1
    else:
        ypred[k] = 0
    
for k in range(len(y_test)):
    y_test[k] = 1 if y_test[k] >= 6.5 else 0

# Finding Error in Prediction
from sklearn.metrics import mean_squared_error as mse
from math import sqrt
error = mse(y_test, ypred)
rmse = sqrt(error)
print("Root Mean Square Error is", rmse)
