# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Real estate.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0])
X[:, 4] = labelencoder.fit_transform(X[:, 4])
X[:, 5] = labelencoder.fit_transform(X[:, 5])

from keras.utils import to_categorical
encoded0 = to_categorical(X[:, 0])
encoded1 = to_categorical(X[:, 4])
encoded2 = to_categorical(X[:, 5])

# Spliting the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Linear Regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train[:, :-1], y_train)

# Predicting the values from the test set
y_pred = regressor.predict(X_test[:, :-1])

from sklearn.metrics import mean_squared_error as mse
from math import sqrt
error = mse(y_test, y_pred)
print("rmse:", sqrt(error))

"""
After checking all variables x1 to x5 are significant
"""