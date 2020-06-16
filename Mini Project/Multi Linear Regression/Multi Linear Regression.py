# Importing Required Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset
data = pd.read_csv('Real estate.csv')
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

# Encoding the Date, Latitude and Longitude
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0])
X[:, 4] = labelencoder.fit_transform(X[:, 4])
X[:, 5] = labelencoder.fit_transform(X[:, 5])

# Scaling the dataset
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1, 1))

# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train[:, :-1], y_train)

# Predicting the values from the test set
y_pred = regressor.predict(X_test[:, :-1])

# Calculating root mean square error for comparision
from sklearn.metrics import mean_squared_error as mse
from math import sqrt
error = mse(y_test, y_pred)
print("rmse:", sqrt(error))

# Plotting test and prediction
plt.scatter(range(83), y_test, color = 'red')
plt.scatter(range(83), y_pred, color = 'blue')
plt.title("Test vs Prediction")
plt.xlabel("Dependent Label"), plt.ylabel("Independent label")
plt.show()


# Building the optimal model using Backward Elimination
import statsmodels.api as sm
X = np.append(arr = np.ones((414, 1)).astype(int), values = X, axis = 1)
X_test = np.append(arr=np.ones((83, 1)).astype(int), values = X_test, axis = 1)

X_opt = X[:, [0, 1, 2, 3, 4, 5, 6]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

"""
Gives same resluts in R**2 as it did in RMSE with x1 to x5
"""

# Predicting the values of the test set using the new OLS regressor
y_OLS = regressor_OLS.predict(exog = X_test[:, :-1])
error = mse(y_test, y_OLS)
print("rmse:", sqrt(error))

# Plotting the backtracking prediction
plt.scatter(range(83), y_test, color = 'red')
plt.scatter(range(83), y_OLS, color = 'blue')
plt.title("Test vs Prediction (Using OLS)")
plt.xlabel("Dependent Label"), plt.ylabel("Independent Label")
plt.show()

# Saving predictions in csv Files
df = pd.DataFrame(y_pred)
df.to_csv("Prediction.csv")
prediction = pd.read_csv("Prediction.csv")

df = pd.DataFrame(y_OLS)
df.to_csv("Prediction (Back Tracking).csv")
prediction = pd.read_csv("Prediction (Back Tracking).csv")