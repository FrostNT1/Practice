# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Country-data.csv')
dictionary = pd.read_csv("data-dictionary.csv")

X = dataset.iloc[:, 1:].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)



# Finding the optimal number of clusters

# Using elbow method to find optimal number of cluster
from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++',
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title("Dendrogram using Ward's Method")
plt.xlabel('Clusters')
plt.ylabel('Euclidean distances')
plt.show()



# By the above Methods we find that 3 is the optimal clustering number

kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 6], X[y_kmeans == 0, 8],
            s = 25, c = 'red', label = 'Label 1', alpha = 0.5)
plt.scatter(X[y_kmeans == 1, 6], X[y_kmeans == 1, 8],
            s = 25, c = 'blue', label = 'Label 2', alpha = 0.5)
plt.scatter(X[y_kmeans == 2, 6], X[y_kmeans == 2, 8],
            s = 25, c = 'green', label = 'Label 3', alpha = 0.5)

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s = 100, c = 'Yellow', label = 'Centroid')

plt.title("Life Expectancy Vs GDP")
plt.xlabel("Life Expectancy")
plt.ylabel("GDP")
plt.legend()
plt.show()

"""
From the various graphs we have plotted, we see that the countries classified
under label 1 needs most help followed by label 3. Label 2 countries seem to
be doing well on their on their own and do not require any assistance 
"""

prior_1 = []
prior_2 = []
prior_3 = []
for i in range(167):
    if y_kmeans[i] == 1:
        prior_1.append(dataset.iloc[i, 0])
    elif y_kmeans[i] == 2:
        prior_2.append(dataset.iloc[i, 0])
    else:
        prior_3.append(dataset.iloc[i, 0])
        
df = pd.DataFrame(prior_1, columns = ["Priority 1"])
df.to_csv("Priority 1.csv", sep = ',', index = False)

df = pd.DataFrame(prior_2, columns = ["Priority 2"])
df.to_csv("Priority 2.csv", sep = ',', index = False)

df = pd.DataFrame(prior_3, columns = ["Priority 3"])
df.to_csv("Priority 3.csv", sep = ',', index = False)
