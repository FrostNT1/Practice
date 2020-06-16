# Importing Libraries
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Dataset
dataset = pd.read_csv("Real estate.csv")
# Taking variables x1 to x5 as shown by RMSE
X = dataset.iloc[:, 1:-2].values
y = dataset.iloc[:, 2].values

# Encoding the dataset accordingly
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0])
X[:, 4] = labelencoder.fit_transform(X[:, 4])

"""
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1, 1))
"""

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Making Regression model and fitting to dataset

# Making Support Vector Regression Model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(random_state = 0)
regressor.fit(X, y)

# Visualizing Regression results
y_pred = regressor.predict(X_test)

plt.scatter(range(83), y_test, color='red', alpha = 0.4)
plt.scatter(range(83), y_pred, color='blue', alpha = 0.4)
plt.title("Test Vs. Prediction (Random Forest Model)")
plt.xlabel("Features")
plt.ylabel("Price")
plt.show()

# Calculating root mean square error
from sklearn.metrics import mean_squared_error as mse
from math import sqrt
error = mse(y_test, y_pred)
print("rmse:", sqrt(error))

"""
The RMSE value is surprisingly low
checking the graph for training set is required too
as there is a high chance of over fitting
"""

plt.scatter(range(331), y_train, color='red', alpha = 0.4)
plt.scatter(range(331), regressor.predict(X_train), color='blue', alpha = 0.4)
plt.title("Train Vs. Prediction (Random Forest Model)")
plt.xlabel("Features")
plt.ylabel("Price")
plt.show()

df = pd.DataFrame(y_pred)
df.to_csv("Prediction.csv")
prediction = pd.read_csv("Prediction.csv")