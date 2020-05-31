# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Train.csv")
X = dataset.iloc[:, :-1:].values
y = dataset.iloc[:, -1].values


# Fitting the Regression Model to the dataset
X = np.append(arr = np.ones((1600, 1)).astype(int), values = X, axis = 1)

X_train = X[:, [0, 1, 2, 3, 4, 5]]

import statsmodels.api as sm
regressor = sm.OLS(endog = y, exog = X_train).fit()
regressor.summary()


# Visualizing the model
y_pred = regressor.predict(exog = X[:, [0, 1, 2, 3, 4, 5]])

actual = plt.scatter(np.array(range(1600)), y, color = 'blue', s = 3.5, alpha = 0.8)
estm = plt.scatter(np.array(range(1600)), y_pred, color = 'orange', s = 3.5, alpha = 0.8)
plt.title("Train vs Prediction(Using OLS)")
plt.xlabel("All Features"), plt.ylabel("Target")
plt.legend((actual, estm), ("Actual Values", "Predicted Values"))
plt.show()

# Multiple feature graph
for i in range(6):
    actual = plt.scatter(X_train[:, i], y, color = 'blue', s = 3.5, alpha = 0.8)
    estm = plt.scatter(X_train[:, i], y_pred, color = 'orange', s = 3.5, alpha = 0.8)
    plt.title("Train vs Prediction(Using OLS)")
    plt.xlabel("Features {}".format(i)), plt.ylabel("Target")
    plt.legend((actual, estm), ("Actual Values", "Predicted Values"))
    plt.show()
    

# Checking Performance of Model
from sklearn.metrics import mean_squared_error as mse
from math import sqrt
error = mse(y, y_pred)
rmse = sqrt(error)

# Predicting new result
obtain = pd.read_csv("Test.csv")
obtain = np.append(arr=np.ones((400, 1)).astype(int), values = obtain, axis = 1)
prediction = regressor.predict(exog = obtain[:, [0,1,2,3,4,5]])

