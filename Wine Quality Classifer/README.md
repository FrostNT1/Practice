# Dataset
Dataset Description:

The dataset consists of certain features of wine in order to predict its quality as good / not good.

Input variables (based on physicochemical tests):
1. Fixed Acidity
2. Volatile Acidity
3. Citric Acid
4. Residual Sugar
5. Chlorides
6. Free Sulfur dioxide
7. Total Sulfur dioxide
8. Density
9. pH
10. Sulphates
11. Alcohol

Output variable (based on sensory data):

12. Quality (Score ==> out of 10)

# Problem Statement
Build a model to classify a given sample of wine as being of Good Quality or Not of Good Quality
(A project in the future can also give the wine a rating out of 10)

# How the Model was made

## Random Forest Classifier
Various classifier models including SVM, Naive Bayes and Random Forest Classifier were tested 
and finally Random Forest Classifier was used to classify the results.

The quality score given was converted to being Good / Not Good by a cut off at 7 rating
then the classifier was used as a binary classifier

## Random Forrest Regression
In contrast to the previous method, the quality was taken as is initially and then using 
random forest the values for the test set were predicted, which were then converted to 1 and 0
using a cut off at 6.5.

Although a very accurate model, it fell short competing to Method 1 and thus, the results in the folder
were acheived through the first method.
