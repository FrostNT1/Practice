# Pulsar Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score as scr
from sklearn.metrics import average_precision_score as pscr
from sklearn.metrics import f1_score

# Importing the dataset
dataset = pd.read_csv('pulsar_stars.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
"""
Commented out as a better fitting Classifier was put in place

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3, p = 2, metric = "minkowski")
classifier.fit(X_train, y_train)
"""

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = "entropy")
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

average_precision = pscr(y_test, y_pred)
print("score is " ,scr(y_test,y_pred))
print('Average Precision Recall score: {0:0.2f}'.format(average_precision))
print("F1 Score = " , f1_score(y_test, y_pred))


df = pd.DataFrame(X_train)
df.to_csv("X Training Set.csv", index = False)

df = pd.DataFrame(y_train)
df.to_csv("Y Training Set.csv", index = False)

df = pd.DataFrame(y_pred)
df.to_csv("Prediction.csv", index = False)
