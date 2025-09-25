# Modeling-the-Likelihood-of-Stroke-from-Health-and-Lifestyle-Predictors

# Data-set
The dataset consists of 5110 observations alongside 12 variables

Description of variables:
- id: Unique idenFfier for each record.
- gender: Gender of the individual (Male, Female, or Other).
- age: Age of the individual (in years).
- hypertension: Whether the individual has hypertension (1 = Yes, 0 = No).
- heart_disease: Whether the individual has a history of heart disease (1 = Yes, 0 = No).
- ever_married: Whether the individual has ever been married (Yes/No).
- work_type: Type of employment (Private, Self-employed, Govt_job, Children, Never_worked).
- residence_type: Type of residency (Urban or Rural).
- avg_glucose_level: Average glucose concentration (mg/dL).
- bmi: Body mass index, calculated as weight in kg divided by the square of height in meters
- (may contain missing values).
- smoking_status: Smoking behavior (formerly smoked, never smoked, smokes, or unknown)
- stroke: Binary variable indicaFng if a stroke has occurred (1 = Yes, 0 = No)

# Introduction
This project will explore the varying mechanisms – demographic, medical, and lifestyle factors (predictors) that risk the likelihood of triggering a stroke, a medical emergency that erupts with disruption to the brain’s blood flow. My variable of interest (dependent variable) is ‘stroke’ , acting binary, turning its value to 1 to signal a stroke has occurred and 0 otherwise. Accordingly, I will adopt the classification approach to designing my study and model to appropriately predict the likelihood of a stroke occurring, identifying the factors that most significantly contribute to this likelihood.

# EDA and pre-processing
I have removed the ‘ID’ and ‘gender_Other’ variable, as it adds no significance to the dataset. Since this is a classification problem, I investigated the dependent variable, ‘stroke’, to examine for any imbalance.

```{r}
# Summary statistics
summary(stroke_data)
str(stroke_data)

# Check for missing values
colSums(is.na(stroke_data))

# Ensure df_diabetes is a tibble (to avoid select() issues)
stroke_data <- as_tibble(stroke_data)

#convert bmi into numeric form
stroke_data$bmi <- as.numeric(as.character(stroke_data$bmi))
colSums(is.na(stroke_data))
str(stroke_data)

#impute missing bmi values 
stroke_data <- stroke_data %>%
  mutate(bmi = if_else(is.na(bmi), median(bmi, na.rm = TRUE), bmi))
colSums(is.na(stroke_data))

#convert categorical variables into factor variables
stroke_data$gender <- as.factor(stroke_data$gender)
stroke_data$ever_married <- as.factor(stroke_data$ever_married)
stroke_data$work_type <- as.factor(stroke_data$work_type)
stroke_data$residence_type <- as.factor(stroke_data$residence_type)
stroke_data$smoking_status <- as.factor(stroke_data$smoking_status)
stroke_data$stroke <- factor(stroke_data$stroke, levels = c(0,1), labels = c("No","Yes"))

#remove gender_Other
if (sum(stroke_data$gender == "Other") <= 1) {
  stroke <- filter(stroke_data, gender != "Other")
  stroke_data$gender <- droplevels(stroke_data$gender)
}

#remove ID variable - not essential to our model 
stroke_data$id <- NULL
```

# Visualisation
```{r}
#GGally plot
GGally::ggpairs(
  stroke_data,
  cardinality_threshold = 500,
  title = "Pairwise Correlations"
)

# Explicitly use dplyr::select() to avoid conflicts
strokedata_numeric <- dplyr::select(stroke_data, -stroke)


# Boxplot for outlier detection --- change into stroke by age? ##needed?
ggplot(stroke_data, aes(y=age, x=stroke)) + 
  geom_boxplot() + ggtitle("Age Levels by Stroke")

# Class distribution (checking for imbalance)
table(stroke_data$stroke)
#plot barchart to visualise imbalance
ggplot(stroke_data, aes(x = factor(stroke), fill = factor(stroke))) +
  geom_bar(stat = "count") +
  labs(title = "Sample distribution of 'Stroke'",
       x = "Stroke (0/1)",
       y = "Count",
       fill = "Stroke") + 
  theme_minimal() +
  scale_fill_discrete(name = "Stroke", labels = c("No Stroke (0)", "Stroke (1)"))
```

#correct for the imbalance in the outcome variable - stroke
```{r}
install.packages("themis")
library(themis)
recipe <- recipe(stroke ~ ., data = stroke_data) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_smote(stroke)
prep_rec <- prep(recipe)
stroke_balanced <- bake(prep_rec, new_data = NULL)
table(stroke_balanced$stroke)

#Test-train split
set.seed(123)
trainIndex <- createDataPartition(stroke_balanced$stroke, p = 0.8, list = FALSE)
train_data <- stroke_balanced[trainIndex, ]
test_data <- stroke_balanced[-trainIndex, ]
table(train_data$stroke)
table(test_data$stroke)

#cross-validation on train_data 
cv <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)
```
Figure 1 shows the large discrepancy between the datapoints in the outcome variable that
classify individuals to have experienced a stroke (4861) and not (249). Lack of datapoints for
stroke will limit the model’s sensitivity towards predicting actual stroke occurrences as this
imbalance impairs the model’s pattern recognition of cases, further restricting its credibility
to generalize to unseen data. I have corrected for this imbalance through oversampling.
To observe the other variables for trends and potential collinearity, I will create a correlogram,
visualizing preliminary interactions in the dataset.

Figure 2 suggests ‘age’, ‘hypertension’, and ‘avg_glucose_level’ show the strongest
associations with ‘stroke’, suggesting they are important variables to be included in the
models designed to predict the likelihood of a stroke occurring. Additionally, categorical
variables, ‘work_type’ and ‘smoking_status’, may appear influential in conjunction with the
other variables. The figure shows weak correlations for ‘gender’, ‘residence_type’, ‘heart
disease’, and ‘bmi’, suggesting they may not contribute a significant impact to the models.
In preparation for fitting classification models, I have adjusted for N/A values and
inconsistencies prevalent in ‘bmi’, imputing the missing values with the group median and
transforming the variable into numeric form.

# Model fitting
To predict the likelihood of a stroke occurring, I began by employing a 10 fold cross validation to access stratified k cross validation and accommodate the imbalanced nature of the dataset and reduce the presence of variance and overfitting that can appear as I formulate potential models. My first model design will be a logistic model, apparent in its full form, utilizing all the predictors in the dataset.

## Model 1 - Logistic Model
```{r}
# (1) Train Logistic Regression Model
logistic_model <- glm(stroke ~ ., data=train_data, family=binomial(logit))
summary(logistic_model)
anova(logistic_model, test="Chisq")

#Predictions
logistic_pred <- predict(logistic_model, newdata=test_data, type="response")
logistic_pred_class <- factor(ifelse(logistic_pred >= 0.5, "Yes", "No"), levels = c("No", "Yes"))
stroke_test_class <- factor(test_data$stroke, levels = c("No", "Yes"))

#Confusion matrix & Accuracy
logistic_model_confusion_matrix <- table(Predicted = logistic_pred_class, Actual = stroke_test_class)
logistic_model_confusion_matrix
accuracy <- function(x) sum(diag(x)) / sum(x) * 100
logistic_model_accuracy <- accuracy(logistic_model_confusion_matrix)
logistic_model_accuracy

#model significance test
install.packages("pscl")
library(pscl)
logistic_G_calc <- logistic_model$null.deviance - logistic_model$deviance
logistic_Gdf <- logistic_model$df.null - logistic_model$df.residual
pscl::pR2(logistic_model)
qchisq(.95, df = logistic_Gdf)
1 - pchisq(logistic_G_calc, logistic_Gdf)

#cross-validation evaluation
train_control <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

logistic_model_cv <- train(
  stroke ~ ., 
  data = train_data,
  method = "glm", 
  family = "binomial",
  metric = "ROC", 
  trControl = train_control
)
print(logistic_model_cv$results)

#ROC curve for logistic model
roc_logistic <- roc(logistic_model_cv$pred$obs, logistic_model_cv$pred$Yes)
plot(roc_logistic, main = "ROC Curve for Logistic Model", print.auc = TRUE, legacy.axes = TRUE)
```
The model presented good fit with the McFadden pseudo-R-squared amounting to 0.33. Complementarily, based on the test data and confusion matrix in table 1, I found the model obtained 77.1% accuracy, 74.4% sensitivity, and 81.7% specificity, indicating the model delivered high predictive accuracy. Figure 4 further displays the model’s strong discriminatory ability highlighting the ROC at 0.853. However, there were multiple variables that did not display impact significance as visible in figure 3 – lack of significance (*** , **), which I will account for by removing them in my next logistic model to gain parsimony.

## Model 2 - Improved Logistic model 
```{r}
logistic_model2 <- glm(stroke ~ . - bmi - gender_Male - gender_Other - ever_married_Yes - 
                         work_type_Govt_job - work_type_Never_worked - residence_type_Urban - smoking_status_smokes - 
                         smoking_status_Unknown, 
                       data = train_data, family = binomial(logit))
summary(logistic_model2)
anova(logistic_model2, test="Chisq")

#Predictions
logistic_pred2 <- predict(logistic_model2, newdata=test_data, type="response")
logistic_pred_class2 <- factor(ifelse(logistic_pred2 >= 0.5, "Yes", "No"), levels = c("No", "Yes"))

#Confusion matrix & Accuracy
logistic_model_confusion_matrix2 <- table(Predicted = logistic_pred_class2, Actual = stroke_test_class)
logistic_model_confusion_matrix2
accuracy <- function(x) sum(diag(x)) / sum(x) * 100
logistic_model_accuracy2 <- accuracy(logistic_model_confusion_matrix2)
logistic_model_accuracy2

#model significance test
logistic2_G_calc <- logistic_model2$null.deviance - logistic_model2$deviance
logistic2_Gdf <- logistic_model2$df.null - logistic_model2$df.residual
pscl::pR2(logistic_model2)
qchisq(.95, df = logistic2_Gdf)
1 - pchisq(logistic2_G_calc, logistic2_Gdf)

#cross-validation evaluation
logistic_model_cv2 <- train(
  stroke ~ . - bmi - gender_Male - gender_Other - ever_married_Yes - 
    work_type_Govt_job - work_type_Never_worked - residence_type_Urban - smoking_status_smokes - 
    smoking_status_Unknown, 
  data = train_data,
  method = "glm", 
  family = "binomial",
  metric = "ROC", 
  trControl = train_control
)
print(logistic_model_cv2$results)

#plot ROC curve
roc_logistic2 <- roc(logistic_model_cv2$pred$obs, logistic_model_cv2$pred$Yes)
plot(roc_logistic2, main = "ROC Curve for Logistic Model(2)", print.auc = TRUE, legacy.axes = TRUE)


##logistic model training 
logistic_model <- train(
  stroke ~ ., data = train_data, method = "glm", family = "binomial",
  metric = "ROC", trControl = cv
)

logistic_model2 <- train(
  stroke ~ . - bmi - gender_Male - gender_Other - ever_married_Yes - 
    work_type_Govt_job - work_type_Never_worked - residence_type_Urban - smoking_status_smokes - 
    smoking_status_Unknown,
  data = train_data, method = "glm", family = "binomial",
  metric = "ROC", trControl = cv
)
```
The reduced logistic model removed the variables that do not display high significance, visible in figure 4 to improve fit and performance. My second model emphasized simplicity that was pronounced with a lower McFadden pseudo-R-squared of 0.327, illustrating a reduction in the model’s explanatory power. On the test dataset, the model displayed 77.8% accuracy, 73.4% sensitivity, and 82.8% specificity, indicating improved predictive accuracy – outperforming the first model. The ROC in figure 5 is identical to the full model and can be interpreted similarly.

## Model 3 - classification tree 
My next model is a classification tree, inspired to capture more complex predictions and capture any non-linear interactions in the dataset. The format of this model supports interpretability and will prove useful in understanding the most significant factors that trigger the likelihood of a stroke occurring. My model includes ‘age’, ‘hypertension’, ‘smoking_status_never.smoked’, ‘residence_type_Urban’ , and ‘heart_disease’ as its key variables, and consists of 11 nodes balancing complexity and interpretability. A concerning remark surrounds the inclusion of ‘residence_type_Urban’ as significant, since it contradicts the direction suggested by the logistic models produced. Nevertheless, the model generates 85.96% classification accuracy, 89.1% sensitivity, 85.1% specificity, and a ROC of 0.930 (Figure 7), making clear its superior performance over prior logistic models. To test for overfitting, I pruned the tree model and found no difference in results suggesting the original classification tree was already optimized and did not inherit any unnecessary complexity.

```{r}
install.packages(tree)
library(tree)
tree_model <- tree(stroke ~., data = train_data)
summary(tree_model)

#plot 
plot(tree_model)
text(tree_model)

#Predict probabilities on test data
tree_pred <- predict(tree_model, newdata = test_data, type = "vector")
#Confusion matrix and accuracy
tree_pred_class <- factor(ifelse(tree_pred[, "Yes"] >= 0.5, "Yes", "No"), levels = c("No", "Yes"))
tree_confusion_matrix <- table(Predicted = tree_pred_class, Actual = stroke_test_class)
tree_confusion_matrix
accuracy <- function(x) sum(diag(x)) / sum(x) * 100
tree_accuracy <- accuracy(tree_confusion_matrix)
tree_accuracy

#cross-validation to find optimal tree size 
cv_stroke <- cv.tree(tree_model, FUN = prune.misclass)
best_size <- cv_stroke$size[which.min(cv_stroke$dev)]
best_size

#prune the classification tree
#we find  the original classification tree is already optimally fitted 
pruned_tree <- prune.misclass(tree_model, best = best_size)
plot(pruned_tree)
text(pruned_tree)

# Predict on test data using the pruned tree and evaluate
pruned_predictions <- predict(pruned_tree, test_data, type = "class")
pruned_confusion_matrix <- table(pruned_predictions, test_data$stroke)
# Calculate pruned accuracy
pruned_accuracy <- sum(diag(pruned_confusion_matrix)) / sum(pruned_confusion_matrix) * 100
pruned_accuracy 

#model significance test
#classification tree uses different fitting criterion - no null deviance and deviance in the glm sense

#cross-validation evaluation
tree_model_cv <- train(
  stroke ~ ., 
  data = train_data,
  method = "rpart", 
  metric = "ROC", 
  trControl = train_control,
  tuneLength = 10  
)
print(tree_model_cv$results)

#ROC plot
roc_tree <- roc(tree_model_cv$pred$obs, tree_model_cv$pred$Yes)
plot(roc_tree, main = "ROC Curve for Classification Tree", print.auc = TRUE, legacy.axes = TRUE)

#train model
tree_model <- train(
  stroke ~ ., 
  data = train_data,
  method = "rpart",          
  metric = "ROC", 
  trControl = cv,
  tuneLength = 10
)
```

## Model 4 - Random forest model

Testing for greater predictive power, I adopted a flexible ensemble method – random forest as my next model. This approach is particularly suitable in handling datasets that involve many predictors and redundant features boasting high applicability to this dataset. Figure 8 displays a rank of variables in order of its importance, highlighting ‘age’, ‘avg_glucose_level’, and ‘hypertension’ at the top. Based on the test data, the model produces 96.97% accuracy, 98.9% sensitivity, 95.1% specificity, and 0.994 ROC (Figure 9). This is now my best performing model, proving most reliable in detecting stroke cases.

```{r}
train_data$stroke <- factor(train_data$stroke, levels = c("No", "Yes"))
test_data$stroke <- factor(test_data$stroke, levels = c("No", "Yes"))
rf_model <- randomForest(stroke ~., data = train_data)
summary(rf_model)

#plot randomforest 
varImpPlot(rf_model)

#Predict probabilities on test data
rf_pred <- predict(rf_model, newdata = test_data, type = "prob")
#Confusion matrix and accuracy
rf_pred_class <- factor(ifelse(rf_pred[, "Yes"] >= 0.5, "Yes", "No"), levels = c("No", "Yes"))
rf_confusion_matrix <- table(Predicted = rf_pred_class, Actual = stroke_test_class)
rf_confusion_matrix
accuracy <- function(x) sum(diag(x)) / sum(x) * 100
rf_accuracy <- accuracy(rf_confusion_matrix)
rf_accuracy

#cross-validation evaluation
rf_model_cv <- train(
  stroke ~ ., 
  data = train_data,
  method = "rf", 
  family = "binomial",
  metric = "ROC", 
  trControl = train_control
)
print(rf_model_cv$results)

#ROC plot
roc_rf <- roc(rf_model_cv$pred$obs, rf_model_cv$pred$Yes)
plot(roc_rf, main = "ROC Curve for random forest", print.auc = TRUE, legacy.axes = TRUE)

#train model
rf_model <- train(
  stroke ~ ., 
  data = train_data,
  method = "rf",             
  metric = "ROC", 
  trControl = cv
)
```

## Model 5 - LDA model
```{r}
lda_model <- lda(stroke ~ . - bmi - gender_Male - gender_Other - ever_married_Yes - 
                    work_type_Govt_job - work_type_Never_worked - residence_type_Urban - smoking_status_smokes - 
                    smoking_status_Unknown, data = train_data)
summary(lda_model)

#confusion matrix and accuracy test 
lda.predict <- predict(lda_model, newdata = test_data)
lda_confusion_matrix <- table(predicted = lda.predict$class, truth = test_data$stroke)
lda_confusion_matrix
accuracy <- function(x) sum(diag(x)) / sum(x) * 100
lda_accuracy <- accuracy(lda_confusion_matrix)
lda_accuracy

#cross-validation evaluation
lda_cv <- train(
  stroke ~ . - bmi - gender_Male - gender_Other - ever_married_Yes - 
    work_type_Govt_job - work_type_Never_worked - residence_type_Urban - smoking_status_smokes - 
    smoking_status_Unknown, 
  data = train_data,
  method = "lda", 
  family = "binomial",
  metric = "ROC", 
  trControl = train_control
)
print(lda_cv$results)

#plot ROC curve
roc_lda <- roc(lda_cv$pred$obs, lda_cv$pred$Yes)
plot(roc_lda, main = "ROC Curve for LDA", print.auc = TRUE, legacy.axes = TRUE)

#train model
lda_model <- train(
  stroke ~ . - bmi - gender_Male - gender_Other - ever_married_Yes - 
    work_type_Govt_job - work_type_Never_worked - residence_type_Urban - smoking_status_smokes - 
    smoking_status_Unknown, 
  data = train_data,
  method = "lda", 
  family = "binomial",
  metric = "ROC", 
  trControl = cv
)
```

## Model 6 - QDA model
```{r}
qda_model <- qda(stroke ~ . - bmi - gender_Male - gender_Other - ever_married_Yes - 
                   work_type_Govt_job - work_type_Never_worked - residence_type_Urban - smoking_status_smokes - 
                   smoking_status_Unknown, data = train_data)
summary(qda_model)

#confusion matrix and accuracy test 
qda.predict <- predict(qda_model, newdata = test_data)
qda_confusion_matrix <- table(predicted = qda.predict$class, truth = test_data$stroke)
qda_confusion_matrix
accuracy <- function(x) sum(diag(x)) / sum(x) * 100
qda_accuracy <- accuracy(qda_confusion_matrix)
qda_accuracy

#cross-validation evaluation
qda_cv <- train(
  stroke ~ . - bmi - gender_Male - gender_Other - ever_married_Yes - 
    work_type_Govt_job - work_type_Never_worked - residence_type_Urban - smoking_status_smokes - 
    smoking_status_Unknown, 
  data = train_data,
  method = "qda", 
  family = "binomial",
  metric = "ROC", 
  trControl = train_control
)
print(qda_cv$results)

#plot ROC curve
roc_qda <- roc(qda_cv$pred$obs, qda_cv$pred$Yes)
plot(roc_qda, main = "ROC Curve for QDA", print.auc = TRUE, legacy.axes = TRUE)

#train model
qda_model <- train(stroke ~ . - bmi - gender_Male - gender_Other - ever_married_Yes - 
                   work_type_Govt_job - work_type_Never_worked - residence_type_Urban - smoking_status_smokes - 
                   smoking_status_Unknown, data = train_data, method = "qda", metric = "ROC", trControl = cv)
```
My last two models serve as baseline analysis, capturing more flexible decision boundaries and evaluating predictive power when the linearity assumption is relaxed.
- LDA: 78.7% accuracy, 70.7% sensitivity, 86.9% specificity, and 0.850 ROC
- QDA: 78.1% accuracy, 73.8% sensitivity, 83.1% specificity, and 0.837 ROC
These findings suggest both models do not match the predictive power of more complex models like random forest. However, they reaffirm the significance of our key predictors, ‘age’, ‘avg_glucose_level’, and ‘hypertension’.

# Model evaluation
```{r}
#Summary
models <- list(
  Full_logistic_model = logistic_model,
  Logistic_model_2= logistic_model2,
  Random_forest = rf_model,
  DecisionTree = tree_model,
  LDA = lda_model,
  QDA = qda_model)

results <- lapply(models, function(m) {
  # Best cross-validated AUC
  best_auc <- max(m$results$ROC, na.rm = TRUE)
  # Confusion matrix on CV predictions
  cm <- confusionMatrix(m$pred$pred, m$pred$obs, positive = "Yes")
  c(AUC = best_auc,
    Sensitivity = cm$byClass["Sensitivity"],
    Specificity = cm$byClass["Specificity"])})

results_df <- do.call(rbind, results) %>%
  as.data.frame() %>%
  rownames_to_column(var = "Model")
View(results_df)
# Display summary in table format
install.packages("knitr")  
library(knitr)
kable(results_df, caption = "Model Performance Summary (AUC, Sensitivity, Specificity)")

#resample summary
#per‐fold metrics
resamp <- resamples(models)
summary(resamp)

#raw per‐fold metrics
vals <- resamp$values
long_vals <- vals %>%
  pivot_longer(
    cols = -Resample,
    names_to  = c("Model", "Metric"),
    names_sep = "~",
    values_to = "Value" )

sd_summary <- long_vals %>%
  group_by(Model, Metric) %>%
  summarize(
    SD = sd(Value),
    .groups = "drop" ) %>%
  arrange(Model, Metric)

sd_wide <- sd_summary %>%
  pivot_wider(
    names_from  = Metric,
    values_from = SD )

print(sd_wide)
```

Overall, all the models illustrated reasonable predictive power, but random forest outperformed the rest of the group highlighting the non-linear relationships and feature interactions prevalent in predicting stroke cases. The reliability and generalizability of the model is further supported by its low standard deviation of 0.00192.
