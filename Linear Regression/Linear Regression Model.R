# Importing Dataset
x_train = read.csv("Linear_X_Train.csv")
y_train = read.csv("Linear_Y_Train.csv")
x_test = read.csv("Linear_X_Test.csv")

regressor = lm(formula=y_train$y ~ x, x_train)

y_pred = predict(regressor, newdata=x_test)

library(ggplot2)
ggplot() + 
  geom_line(aes(x = x_test$x, y = predict(regressor, newdata=x_test)),
            color='orange') +
  xlab("Time Spent") +
  ylab("Marks Attained") +
  ggtitle("Time Spent Vs. Marks Scored")

write.csv(y_pred, "y_test.csv", row.names = FALSE)