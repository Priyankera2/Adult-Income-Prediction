data=read.csv(file.choose())
View(data)
str(data)
summary(data)
# Convert income(target variable) to a factor
data$income <- as.factor(data$income)
int_columns <- sapply(data, is.integer)
data[int_columns]<-lapply(data[int_columns], as.numeric)
char_columns <- sapply(data, is.character)
data[char_columns] <- lapply(data[char_columns], as.factor)
data[char_columns] <- lapply(data[char_columns], as.factor)
str(data)
#splitting the dataset
library(caTools)
set.seed(123)
split<-sample.split(data$income,SplitRatio=.7)
train_dataset=subset(data,split==TRUE)
test_dataset=subset(data,split==FALSE)
str(train_dataset)
# Prepare training and test features (excluding the target variable)
train_features <- train_dataset[, -15]
test_features <- test_dataset[, -15]
# Prepare the target variables (income)
train_target <- train_dataset$income
test_target <- test_dataset$income
# Should return 0 if no missing values
str(train_features)
str(test_features)
# Convert factor columns to dummy variables using model.matrix
train_features <- model.matrix(~ . - 1, data = train_features)
test_features <- model.matrix(~ . - 1, data = test_features)


library(class)

# Train the KNN model
k <- 3  
knn_model <- knn(train_features, test_features, train_target, k)

(table(knn_model, test_target))
# Standardize numeric features
train_features <- scale(train_features)
test_features <- scale(test_features)
# Create a confusion matrix to evaluate the KNN model
confusion_matrix <- table(Predicted = knn_model, Actual = test_target)

# Print confusion matrix
print(confusion_matrix)

# Calculate accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("Accuracy of the KNN model:", accuracy*100, "\n")
error_rate <- 1 - sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("Error Rate of the KNN model:", error_rate, "\n")

# Extract values from confusion matrix
true_positive <- confusion_matrix[2, 2]  # Predicted '>50K' and Actual '>50K'
false_positive <- confusion_matrix[1, 2]  # Predicted '>50K' and Actual '<=50K'
false_negative <- confusion_matrix[2, 1]  # Predicted '<=50K' and Actual '>50K'
true_negative <- confusion_matrix[1, 1]  # Predicted '<=50K' and Actual '<=50K'

# Precision calculation with check for zero denominator
if ((true_positive + false_positive) == 0) {
  precision <- 0  # Set to 0 if denominator is 0
} else {
  precision <- true_positive / (true_positive + false_positive)
}
cat("Precision:", precision, "\n")

# Recall calculation with check for zero denominator
if ((true_positive + false_negative) == 0) {
  recall <- 0  # Set to 0 if denominator is 0
} else {
  recall <- true_positive / (true_positive + false_negative)
}
cat("Recall:", recall, "\n")

# F1 Score calculation with check for zero denominator
if ((precision + recall) == 0) {
  f1_score <- 0  # Set to 0 if denominator is 0
} else {
  f1_score <- 2 * (precision * recall) / (precision + recall)
}
cat("F1 Score:", f1_score, "\n")


#*****************************NAIVE BAYES*********************************
# Load the e1071 package (Naive Bayes classifier)
library(e1071)

# Train the Naive Bayes model
nb_model <- naiveBayes(train_features, train_target)

# Print the model summary
print(nb_model)
# Predict using the Naive Bayes model
nb_predictions <- predict(nb_model, test_features)

# View predictions
head(nb_predictions)
# Load caret package for confusion matrix and performance metrics
library(caret)

# Generate confusion matrix
conf_matrix <- confusionMatrix(nb_predictions, test_target)

# Print confusion matrix and performance metrics
print(conf_matrix)
accuracy <- conf_matrix$overall["Accuracy"]
cat("Accuracy: ", accuracy*100, "\n")
# Extract precision, recall, and F1 score
precision <- posPredValue(nb_predictions, test_target)
recall <- sensitivity(nb_predictions, test_target)
f1_score <- (2 * precision * recall) / (precision + recall)

# Print the metrics
cat("Precision: ", precision, "\n")
cat("Recall: ", recall, "\n")
cat("F1 Score: ", f1_score, "\n")
# Calculate error rate
error_rate <- mean(nb_predictions != test_target)
cat("Error Rate: ", error_rate, "\n")
#*******************DECISSION TREE********************
library(rpart)
library(rpart.plot)
# Train a Decision Tree model
dt_model <- rpart(income ~ ., data = train_dataset, method = "class")

# Check the model summary
summary(dt_model)
# Make predictions
dt_predictions <- predict(dt_model, test_dataset, type = "class")

# Check the first few predictions
head(dt_predictions)
# Load necessary libraries for metrics
library(caret)
library(e1071)

# Calculate the confusion matrix
conf_matrix_dt <- confusionMatrix(dt_predictions, test_target)

# Print the confusion matrix and accuracy
print(conf_matrix_dt)

# Calculate and print the error rate
error_rate_dt <- 1 - conf_matrix_dt$overall['Accuracy']
cat("Error Rate: ", error_rate_dt, "\n")


# Extract and print the accuracy from the confusion matrix
accuracy_dt <- conf_matrix_dt$overall['Accuracy']
cat("Accuracy: ", accuracy_dt*100, "\n")

library(MLmetrics)  # For better handling of metrics

# Confusion Matrix
conf_matrix_dt <- confusionMatrix(dt_predictions, test_target)
print(conf_matrix_dt)

# Calculate Precision, Recall, and F1-Score using MLmetrics (it handles smoothing)
precision_dt <- Precision(dt_predictions, test_target)
recall_dt <- Recall(dt_predictions, test_target)
f1_dt <- F1_Score(dt_predictions, test_target)

# Print the metrics
cat("Precision: ", precision_dt, "\n")
cat("Recall: ", recall_dt, "\n")
cat("F1-Score: ", f1_dt, "\n")

rpart.plot(dt_model, type = 3, extra = 1, fallen.leaves = TRUE)
#************************NEURAL NET*************************************
# Convert the target variable to numeric (0 for <=50K, 1 for >50K)
train_dataset$income <- as.numeric(train_dataset$income) - 1
test_dataset$income <- as.numeric(test_dataset$income) - 1

# Convert all factor columns (except the target) to numeric in both train and test datasets
factor_columns <- sapply(train_dataset, is.factor)
train_dataset[factor_columns] <- lapply(train_dataset[factor_columns], as.numeric)
test_dataset[factor_columns] <- lapply(test_dataset[factor_columns], as.numeric)

# Now, train the neural network model
library(neuralnet)
nn_model <- neuralnet(income ~ ., data = train_dataset, hidden = c(5, 3), linear.output = FALSE)

# Make predictions on the test dataset
nn_pred <- predict(nn_model, test_dataset, type = "class")

# Evaluate the accuracy
conf_matrix <- table(Predicted = nn_pred, Actual = test_dataset$income)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
print(paste("Accuracy: ", accuracy*100))
# Calculate Precision, Recall, F1 Score, and Error Rate
TP <- conf_matrix[2, 2]  # True Positives
TN <- conf_matrix[1, 1]  # True Negatives
FP <- conf_matrix[1, 2]  # False Positives
FN <- conf_matrix[2, 1]  # False Negatives

# Precision
precision <- TP / (TP + FP)
# Recall
recall <- TP / (TP + FN)
# F1 Score
f1_score <- 2 * (precision * recall) / (precision + recall)
# Error Rate
error_rate <- (FP + FN) / sum(conf_matrix)

# Print the results
print(paste("Precision: ", precision))
print(paste("Recall: ", recall))
print(paste("F1 Score: ", f1_score))
print(paste("Error Rate: ", error_rate))
