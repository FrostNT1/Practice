# Clustering The Countries By Using Unsupervised Learning For HELP International

## Objective:
To categorise the countries using socio-economic and health factors that determine the overall
development of the country.

## About organization:
HELP International is an international humanitarian NGO that is committed to fighting poverty and
providing the people of backward countries with basic amenities and relief during the time of
disasters and natural calamities.

## Problem statement:
HELP International have been able to raise around $ 10 million. Now the CEO of the NGO needs to
decide how to use this money strategically and effectively. So, CEO has to make decision to choose
the countries that are in the direst need of aid. Hence, your Job as a Data scientist is to categorise
the countries using some socio-economic and health factors that determine the overall development
of the country. Then you need to suggest the countries which the CEO needs to focus on the most.

# Solution

## Method used:
The approach was to group together the countries in various clusters by their statistics provided
using K-Means (K-Means ++) algorithm. To find the optimal number of clusters for the data, Dendrogram
was also used parallel to using the Elbow Method.

## Visualization 
Upon finding the optimal value of clusters, the values were assigned and the clusters were made. Then
using matplotlib, the clusters were mapped onto a graph comparing a few statistics to see which clusters
were the ones that required the fundings the most.

## Final output
According to the found correlation the priorites were assigned to the countries and were saved in CSV files.
