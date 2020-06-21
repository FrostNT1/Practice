# Importing the libraries
import pandas as pd

# Importing the dataset
X_train = pd.read_csv('XTrain.csv')
X_train = X_train.iloc[:, [0,1,2,4,5,6,7]].values
X_test = pd.read_csv("XTest.csv")
X_test = X_test.iloc[:, [0,1,2,4,5,6,7]].values
y_train = pd.read_csv("YTrain.csv")
y_train = y_train.iloc[:, :].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting Classifier Regression
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3, p = 2, metric = "minkowski")
classifier.fit(X_train, y_train.ravel())

# Predicting test set using classifier
y_pred = classifier.predict(X_train)

# Making Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_pred)

y_pred = classifier.predict(X_test)
df = pd.DataFrame(y_pred)
df.to_csv("Prediction.csv", index = False)