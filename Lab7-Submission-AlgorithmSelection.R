# CLASSIFICATION ----
## Installing and loading packages ====
if (require("stats")) {
  require("stats")
} else {
  install.packages("stats", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## mlbench 
if (require("mlbench")) {
  require("mlbench")
} else {
  install.packages("mlbench", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## caret 
if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## MASS 
if (require("MASS")) {
  require("MASS")
} else {
  install.packages("MASS", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## glmnet 
if (require("glmnet")) {
  require("glmnet")
} else {
  install.packages("glmnet", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## e1071 
if (require("e1071")) {
  require("e1071")
} else {
  install.packages("e1071", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## kernlab 
if (require("kernlab")) {
  require("kernlab")
} else {
  install.packages("kernlab", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## rpart 
if (require("rpart")) {
  require("rpart")
} else {
  install.packages("rpart", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## A. Linear Algorithms ----
### 1a. Logistic Regression without caret ----
# Loading and splitting the dataset
library(readr)
Loan_Default <- read_csv("data/Loan_Default.csv", 
    col_types = cols(Employed = col_factor(levels = c("1", 
        "0")), Default = col_factor(levels = c("1", 
        "0"))))
View(Loan_Default) 

# An 80:20 split of the dataset
train_index <- createDataPartition(Loan_Default$Default,
                                   p = 0.8,
                                   list = FALSE)
Loan_Default_train <- Loan_Default[train_index, ]
Loan_Default_test <- Loan_Default[-train_index, ]

#Training the model
Loan_Default_model_glm <- glm(Default ~ ., data = Loan_Default_train,
                          family = binomial(link = "logit"))

# Displaying the model's details
print(Loan_Default_model_glm)

# Making predictions on the test data
probabilities <- predict(Loan_Default_model_glm, Loan_Default_test, type = "response")
print(probabilities)
# A probability greater than 0.5 = 1(default), less than 0.5 = 0(non-default)
predictions <- ifelse(probabilities > 0.5, 1, 0)
print(predictions)
# Displaying the model's evaluation metrics
table(predictions, Loan_Default_test$Default)

### 1b. Logistic Regression Using caret ----
# Since we've already loaded and split the dataset, we go straight to training the model
# Applying 10-fold cross validation resampling method
train_control <- trainControl(method = "cv", number = 10)
set.seed(7)
Loan_Default_caret_model_logistic <-
  train(Default ~ ., data = Loan_Default_train,
        method = "regLogistic", metric = "Accuracy",
        preProcess = c("center", "scale"), trControl = train_control)
# Displaying the model
print(Loan_Default_caret_model_logistic)
# Make Predictions
predictions <- predict(Loan_Default_caret_model_logistic,
                       Loan_Default_test[, 1:4])
# Displaying the model's evaluation metrics
confusion_matrix <-
  caret::confusionMatrix(predictions,
                         Loan_Default_test[, 1:5]$Default)
print(confusion_matrix)

fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")

### 2a.Linear Discriminant Analysis without caret ----
# Training the model
Loan_default_model_lda <- lda(Default ~ ., data = Loan_Default_train)
# Displaying the model
print(Loan_default_model_lda)
# Making predictions
predictions <- predict(Loan_default_model_lda,
                       Loan_Default_test[, 1:4])$class
# Display Model's evaluation metrics
table(predictions, Loan_Default_test$Default)

### 2b. Linear Discriminant Analysis using caret ----
# Train the model
set.seed(10)
## applying Leave One Out Cross Validation resampling method
train_control <- trainControl(method = "LOOCV")
Loan_default_caret_model_lda <- train(Default ~ .,
                                  data = Loan_Default_train,
                                  method = "lda", metric = "Accuracy",
                                  preProcess = c("center", "scale"),
                                  trControl = train_control)
# Display the model's details
print(Loan_default_caret_model_lda)

# Making predictions on the test dataset
predictions <- predict(Loan_default_caret_model_lda,
                       Loan_Default_test[, 1:4])

# Display the model's evaluation metrics 
confusion_matrix <-
  caret::confusionMatrix(predictions,
                         Loan_Default_test[, 1:5]$Default)
print(confusion_matrix)

fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")

### 3a. Regularized Linear Regression without caret ----
# Splitting to get the feature matrix and target matrix
x <- as.matrix(Loan_Default[, 1:4])
y <- as.matrix(Loan_Default[, 5])

# Training the model; using elastic net 
Loan_default_model_glm <- glmnet(x, y, family = "binomial",
                             alpha = 0.5, lambda = 0.001)

# Displaying the model's details 
print(Loan_default_model_glm)

# Making predictions 
predictions <- predict(Loan_default_model_glm, x, type = "class")

# Displaying the model's evaluation metrics 
table(predictions, Loan_Default$Default)

### 3b. Regularized Linear Regression using caret ----
# Training the model
set.seed(7)

# Resampling using 10 fold cross validation
train_control <- trainControl(method = "cv", number = 10)
Loan_default_caret_model_glmnet <-
  train(Default ~ ., data = Loan_Default_train,
        method = "glmnet", metric = "Accuracy",
        preProcess = c("center", "scale"), trControl = train_control)

#Display the model
print(Loan_default_caret_model_glmnet)

# Make predictions
predictions <- predict(Loan_default_caret_model_glmnet,
                       Loan_Default_test[, 1:4])

# Display the model's evaluation metrics
confusion_matrix <-
  caret::confusionMatrix(predictions,
                         Loan_Default_test[, 1:5]$Default)
print(confusion_matrix)

fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")


## B. Non-Linear Algorithms ----
### 1. CART - Decision trees without caret ----
#Training the model
Loan_default_model_rpart <- rpart(Default ~ ., data = Loan_Default_train)

#Displaying model
print(Loan_default_model_rpart)

#Making predictions using the test dataset
predictions <- predict(Loan_default_model_rpart,
                       Loan_Default_test[, 1:4],
                       type = "class")

#Displaying the evaluation metrics
table(predictions, Loan_Default_test$Default)

confusion_matrix <-
  caret::confusionMatrix(predictions,
                         Loan_Default_test[, 1:5]$Default)
print(confusion_matrix)

fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")



### 2. Naïve Bayes Classifier without Caret
#Training the model
Loan_default_model_nb <- naiveBayes(Default ~ .,
                                data = Loan_Default_train)

#Displaying the model's details
print(Loan_default_model_nb)

#Making predictions
predictions <- predict(Loan_default_model_nb,
                       Loan_Default_test[, 1:4])

# Displaying the evaluation metrics
confusion_matrix <-
  caret::confusionMatrix(predictions,
                         Loan_Default_test[, 1:5]$Default)
print(confusion_matrix)

fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")




### 3. kNN using caret ----
# Training the model
set.seed(7)
# Resampling using 10 - fold cross validation
train_control <- trainControl(method = "cv", number = 10)
Loan_default_caret_model_knn <- train(Default ~ ., data = Loan_Default,
                                  method = "knn", metric = "Accuracy",
                                  preProcess = c("center", "scale"),
                                  trControl = train_control)
#Displaying the model
print(Loan_default_caret_model_knn)

# Making predictions
predictions <- predict(Loan_default_caret_model_knn,
                       Loan_Default_test[, 1:4])

# Displaying evaluation metrics
confusion_matrix <-
  caret::confusionMatrix(predictions,
                         Loan_Default_test[, 1:5]$Default)
print(confusion_matrix)

fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")


### 4a. Support Vector Machine without CARET ----
# Training the model 
Loan_default_model_svm <- ksvm(Default ~ ., data = Loan_Default_train,
                           kernel = "rbfdot")

#Displaying the model
print(Loan_default_model_svm)

# Make predictions 
predictions <- predict(Loan_default_model_svm, Loan_Default_test[, 1:4],
                       type = "response")

# Displaying the evaluation metrics 
table(predictions, Loan_Default_test$Default)

confusion_matrix <-
  caret::confusionMatrix(predictions,
                         Loan_Default_test[, 1:5]$Default)
print(confusion_matrix)

fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")

### 4b. Support Vector Machine using CARET ----
# Training the model 
set.seed(7)
# Resampling using 10-fold cross validation
train_control <- trainControl(method = "cv", number = 10)
Loan_default_caret_model_svm_radial <- 
  train(Default ~ ., data = Loan_Default_train, method = "svmRadial",
        metric = "Accuracy", trControl = train_control)

# Display the model
print(Loan_default_caret_model_svm_radial)

# Making predictions 
predictions <- predict(Loan_default_caret_model_svm_radial,
                       Loan_Default_test[, 1:4])

# Display the evaluation metrics 
table(predictions, Loan_Default_test$Default)
confusion_matrix <-
  caret::confusionMatrix(predictions,
                         Loan_Default_test[, 1:5]$Default)
print(confusion_matrix)

fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")




















 


















































