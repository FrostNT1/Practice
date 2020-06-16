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

# Feature Scaling the dataset
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1, 1))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Making Regression model and fitting to dataset

# Making Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg =  PolynomialFeatures(degree = 7)
X_train = poly_reg.fit_transform(X_train)
X_test = poly_reg.fit_transform(X_test)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Visualizing Regression results
y_pred = lin_reg.predict(X_test)

plt.scatter(range(83), y_test, color='red', alpha = 0.4)
plt.scatter(range(83), y_pred, color='blue', alpha = 0.4)
plt.title("Test Vs. Prediction (Polynomial Regression Model)")
plt.xlabel("Features")
plt.ylabel("Price")
plt.show()

# Calculating root mean square error
from sklearn.metrics import mean_squared_error as mse
from math import sqrt
error = mse(y_test, y_pred)
print("rmse:", sqrt(error))
# Upon testing degree 6 and 7 are the most accurate without being over fitted

df = pd.DataFrame(y_pred)
df.to_csv("Prediction.csv")
prediction = pd.read_csv("Prediction.csv")