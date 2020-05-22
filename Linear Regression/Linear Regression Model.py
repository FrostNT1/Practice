# Importing Libraries to be used in making model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing datasets
x_train = pd.read_csv("Linear_X_Train.csv")
y_train = pd.read_csv("Linear_Y_Train.csv")
x_test = pd.read_csv("Linear_X_Test.csv")

# Creating Linear Regression Model using scikit learn and fitting to data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Using the model made to predict on given values of test
y_pred = regressor.predict(x_test)

# Making graph for visualization
plt.scatter(x_train, y_train, color = 'green')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title("Time Spent Vs. Marks Scored")
plt.xlabel("Time spent"), plt.ylabel("Marks Attained")
plt.show()
