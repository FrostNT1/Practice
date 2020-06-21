# Importing the libraries
import pandas as pd

# Importing the dataset
X = pd.read_csv('XTrain.csv')
X = X.iloc[:, [0,1,2,4,5,6,7]].values
y = pd.read_csv("YTrain.csv")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting Classifier Regression
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, p = 2, metric = "minkowski")
classifier.fit(X_train, y_train)

# Predicting test set using classifier
y_pred = classifier.predict(X_test)

# Making Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score as s
print(s(y_test,y_pred))

test = pd.read_csv("Xtest.csv")
y_pred = classifier.predict(test.iloc[:, [0,1,2,4,5,6,7]].values)
df = pd.DataFrame(y_pred)
df.to_csv("Prediction.csv", index = False)