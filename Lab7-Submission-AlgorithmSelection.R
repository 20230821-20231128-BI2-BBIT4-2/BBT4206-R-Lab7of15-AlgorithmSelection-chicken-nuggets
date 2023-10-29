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


# ASSOCIATION ----
# STEP 1. Install and Load the Required Packages ----
## arules ----
if (require("arules")) {
  require("arules")
} else {
  install.packages("arules", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## arulesViz ----
if (require("arulesViz")) {
  require("arulesViz")
} else {
  install.packages("arulesViz", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## tidyverse ----
if (require("tidyverse")) {
  require("tidyverse")
} else {
  install.packages("tidyverse", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## readxl ----
if (require("readxl")) {
  require("readxl")
} else {
  install.packages("readxl", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## knitr ----
if (require("knitr")) {
  require("knitr")
} else {
  install.packages("knitr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## ggplot2 ----
if (require("ggplot2")) {
  require("ggplot2")
} else {
  install.packages("ggplot2", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## lubridate ----
if (require("lubridate")) {
  require("lubridate")
} else {
  install.packages("lubridate", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## plyr ----
if (require("plyr")) {
  require("plyr")
} else {
  install.packages("plyr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## dplyr ----
if (require("dplyr")) {
  require("dplyr")
} else {
  install.packages("dplyr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## naniar ----
if (require("naniar")) {
  require("naniar")
} else {
  install.packages("naniar", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## RColorBrewer ----
if (require("RColorBrewer")) {
  require("RColorBrewer")
} else {
  install.packages("RColorBrewer", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

# STEP 2. Load and pre-process the dataset ----
## FORMAT 1: Single Format ----
# An example of the single format transaction data is presented
# here: "data/transactions_single_format.csv" and loaded as follows:
transactions_single_format <-
  read.transactions("data/transactions_single_format.csv",
                    format = "single", cols = c(1, 2))

View(transactions_single_format)
print(transactions_single_format)

# Reading the set
retail <- read_excel("data/GroceryStoreDataSet.csv")
dim(retail)

### Handle missing values ----
# Are there missing values in the dataset?
any_na(retail)

# How many?
n_miss(retail)

# What is the proportion of missing data in the entire dataset?
prop_miss(retail)

# What is the number and percentage of missing values grouped by
# each variable?
miss_var_summary(retail)

# Which variables contain the most missing values?
gg_miss_var(retail)

# Which combinations of variables are missing together?
gg_miss_upset(retail)

#### Removing the observations with missing values ----
retail_removed_obs <- retail %>% filter(complete.cases(.))

# We end up with 406,829 observations to create the association rules
# instead of the initial 541,909 observations.
dim(retail_removed_obs)

# Are there missing values in the dataset?
any_na(retail_removed_obs)

## Record only the `items` variable ----
# Notice that at this point, the single format ensures that each transaction is
# recorded in a single observation. We therefore no longer require the
# `invoice_no` variable and all the other variables in the data frame except
# the itemsets.

### OPTION 1 ----
transaction_data <-
  transaction_data %>%
  dplyr::select("items" = V1)
#  %>% mutate(items = paste("{", items, "}", sep = ""))

View(transaction_data)

## Save the transactions in CSV format ----
### OPTION 1 ----
write.csv(transaction_data,
          "data/transactions_basket_format_grocery_store.csv",
          quote = FALSE, row.names = FALSE)

## Read the transactions from the CSV file ----
### OPTION 1 ----
tr <-
  read.transactions("data/transactions_basket_format_grocery_store.csv",
    format = "basket",
    header = TRUE,
    rm.duplicates = TRUE,
    sep = ","
  )

print(tr)
summary(tr)

# STEP 3. Create the association rules ----
association_rules_prod_name <- apriori(tr,
                                       parameter = list(support = 0.01,
                                                        confidence = 0.8,
                                                        maxlen = 10))

# STEP 3. Print the association rules ----
## OPTION 1 ----
# Threshold values of support = 0.01, confidence = 0.8, and
# maxlen = 10 results in a total of 83 rules when using the
# stock code to identify the products.
summary(association_rules)
inspect(association_rules)
# To view the top 10 rules
inspect(association_rules[1:10])
plot(association_rules)

### Remove redundant rules ----
# We can remove the redundant rules as follows:
subset_rules <-
  which(colSums(is.subset(association_rules,
                          association_rules)) > 1)
length(subset_rules)
association_rules_no_reps <- association_rules[-subset_rules]

# This results in 40 non-redundant rules (instead of the initial 83 rules)
summary(association_rules_no_reps)
inspect(association_rules_no_reps)

write(association_rules_no_reps,
      file = "rules/association_rules_grocery.csv")



















 


















































