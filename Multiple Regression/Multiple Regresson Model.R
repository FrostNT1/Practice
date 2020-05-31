# Importing dataset

train = read.csv("Train.csv")

# Creating regressor and fitting it

regressor = lm(formula = target ~ .,
               data = train)

summary(regressor)

# using backward elimination to create optimized model
# No need for elemination of features

# Predictiong Values

test = read.csv("Test.csv")
prediction = predict(regressor, newdata = test)
