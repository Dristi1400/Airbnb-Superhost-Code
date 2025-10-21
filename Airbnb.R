# 1. Load Required Libraries
library(tidyverse)
library(caret)
library(randomForest)
library(xgboost)
library(e1071)
library(ROCR)
library(corrplot)
library(Matrix)

# 2. Set Working Directory & Load CSV File
#setwd("C:/Users/lavan/Downloads")
data <- read.csv("airbnb_ratings_new.csv", stringsAsFactors = FALSE)

# 3.corrected: Create binary Superhost column from logical TRUE/FALSE
data$host_is_superhost <- as.integer(data$Host.Is.Superhost)

# 4. Clean 'Price' column (capital P)
data$Price <- as.numeric(gsub("[$,]", "", data$Price))

# 5. EDA — Superhost Distribution
summary(data)
table(data$host_is_superhost, useNA = "ifany")

# 6. Histogram of number of reviews by Superhost status
ggplot(data %>%
         filter(!is.na(host_is_superhost),
                !is.na(Number.of.reviews),
                Number.of.reviews <= 100),
       aes(x = Number.of.reviews, fill = factor(host_is_superhost))) +
  geom_histogram(position = "dodge", bins = 50, alpha = 0.8) +
  labs(title = "Side-by-Side Reviews by Superhost Status (≤ 100 Reviews)",
       x = "Number of Reviews",
       fill = "Superhost") +
  theme_minimal()
# Log-Scale Histogram for Number of Reviews

ggplot(data %>%
         filter(!is.na(host_is_superhost),
                !is.na(Number.of.reviews),
                Number.of.reviews > 0),  # Exclude zero reviews to apply log scale
       aes(x = Number.of.reviews, fill = factor(host_is_superhost))) +
  geom_histogram(position = "identity", bins = 50, alpha = 0.6) +
  scale_x_log10() +
  labs(title = "Log-Scale Reviews by Superhost Status",
       x = "Number of Reviews (log scale)",
       fill = "Superhost") +
  theme_minimal()

# 7. Correlation heatmap
# Load required libraries
library(corrplot)
library(dplyr)

# 1. Select variables relevant to Superhost prediction
vars <- c(
  "host_is_superhost",
  "Host.total.listings.count",
  "Price",
  "Availability.365",
  "Number.of.reviews",
  "Reviews.per.month",
  "Review.Scores.Rating",
  "Review.Scores.Cleanliness",
  "Review.Scores.Communication",
  "Review.Scores.Checkin",
  "Review.Scores.Location",
  "Review.Scores.Value",
  "Bathrooms", "Bedrooms", "Accommodates"
)

# 2. Clean and prepare numeric data
corr_data <- data[, vars] %>%
  na.omit() %>%
  mutate(across(everything(), ~ suppressWarnings(as.numeric(.)))) %>%
  select(where(~ sd(., na.rm = TRUE) > 0))

# 3. Compute correlation matrix
cor_matrix <- cor(corr_data, use = "complete.obs")

# 4. Plot a clean, RStudio-friendly correlation heatmap
# Clean and readable correlation plot
corrplot(cor_matrix,
         method = "circle",                        # Use circle for clarity
         type = "upper",                           # Show only upper triangle
         order = "hclust",                         # Group similar variables
         tl.cex = 0.7,                             # Label font size
         tl.col = "black",                         # Label color
         tl.srt = 45,                              # Rotate labels
         addCoef.col = NA,                         # Disable correlation numbers
         number.cex = 0.7,                         # (no effect here since addCoef.col = NA)
         col = colorRampPalette(c("red", "white", "blue"))(200),
         mar = c(1, 1, 1, 1))                      # Tighter margins
###  Top 5 Variables Correlated with host_is_superhost
# Get sorted correlation values with host_is_superhost
cor_host <- cor_matrix["host_is_superhost", ]
cor_host <- sort(abs(cor_host), decreasing = TRUE)

# Remove self-correlation
cor_host <- cor_host[names(cor_host) != "host_is_superhost"]

# Print top 5
top_5 <- head(cor_host, 5)
print(top_5)

# Create a named vector with top correlations
top_cor <- c(
  Number.of.reviews = 0.2622633,
  Reviews.per.month = 0.2504184,
  Review.Scores.Cleanliness = 0.1639982,
  Review.Scores.Value = 0.1541301,
  Review.Scores.Rating = 0.1531889
)

# Convert to data frame
top_cor_df <- data.frame(
  Variable = names(top_cor),
  Correlation = as.numeric(top_cor)
)

# Load ggplot2 (if not already loaded via tidyverse)
library(ggplot2)

# Create the bar plot
ggplot(top_cor_df, aes(x = reorder(Variable, Correlation), y = Correlation)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 5 Features Correlated with Superhost Status",
       x = "Feature",
       y = "Absolute Correlation") +
  theme_minimal(base_size = 13)

# 8. FEATURE ENGINEERING

# A. reviews_per_day_available
# → Indicates how frequently the host receives reviews relative to how often the listing is available.
data$reviews_per_day_available <- with(data, Reviews.per.month / (Availability.365 / 30 + 1))

# B. total_rooms
# → Total number of rooms (bedrooms + bathrooms) as a proxy for listing size.
data$total_rooms <- with(data, Bedrooms + Bathrooms)

# C. accommodates_per_room
# → How efficiently the space is used; higher values might suggest tight/compact accommodations.
data$accommodates_per_room <- with(data, Accommodates / (total_rooms + 1))

# D. avg_review_score
# → Average of all available review scores to represent overall guest satisfaction.
review_vars <- c("Review.Scores.Rating",
                 "Review.Scores.Cleanliness",
                 "Review.Scores.Checkin",
                 "Review.Scores.Communication",
                 "Review.Scores.Location",
                 "Review.Scores.Value")

# Convert all review score columns to numeric in case they are characters/factors
data[review_vars] <- lapply(data[review_vars], function(x) as.numeric(as.character(x)))
data$avg_review_score <- rowMeans(data[, review_vars], na.rm = TRUE)

# E. is_experienced_host
# → A binary flag indicating whether the host has more than 5 listings (suggesting experience).
data$is_experienced_host <- ifelse(data$Host.total.listings.count > 5, 1, 0)

# F. value_per_dollar
# → Measures how well-rated the listing's value is in relation to its price.
data$value_per_dollar <- data$Review.Scores.Value / (data$Price + 1)

# OPTIONAL: View a summary of engineered features
engineered_vars <- c("reviews_per_day_available",   # A
                     "total_rooms",                 # B
                     "accommodates_per_room",       # C
                     "avg_review_score",            # D
                     "is_experienced_host",         # E
                     "value_per_dollar")            # F

summary(data[engineered_vars])

# Correlation of Top Features (Original + Engineered)

# Fix: Load tidyverse before anything
library(tidyverse)

# Step 1: Define Variables
top_5_raw <- c("Number.of.reviews", "Reviews.per.month",
               "Review.Scores.Cleanliness", "Review.Scores.Value", "Review.Scores.Rating")

engineered_vars <- c("reviews_per_day_available", "total_rooms", "accommodates_per_room",
                     "avg_review_score", "is_experienced_host", "value_per_dollar")

corr_vars <- c("host_is_superhost", top_5_raw, engineered_vars)

# Step 2: Prepare Clean Data
corr_data <- data %>%
  dplyr::select(all_of(corr_vars)) %>%
  mutate(across(everything(), ~ suppressWarnings(as.numeric(.)))) %>%
  na.omit()


# Step 3: Correlation Matrix
cor_matrix <- cor(corr_data, use = "complete.obs")
cor_to_superhost <- sort(abs(cor_matrix["host_is_superhost", -1]), decreasing = TRUE)

# Step 4: Format for Plot
cor_df <- data.frame(
  Feature = names(cor_to_superhost),
  Correlation = as.numeric(cor_to_superhost),
  Type = ifelse(names(cor_to_superhost) %in% engineered_vars, "Engineered", "Original")
)

# Step 5: Plot
ggplot(cor_df, aes(x = reorder(Feature, Correlation), y = Correlation, fill = Type)) +
  geom_bar(stat = "identity", alpha = 0.9) +
  coord_flip() +
  labs(title = "Correlation with Superhost Status: Original vs Engineered Features",
       x = "Feature", y = "Absolute Correlation") +
  scale_fill_manual(values = c("Original" = "steelblue", "Engineered" = "tomato")) +
  theme_minimal(base_size = 13)


# =========================
# Airbnb Superhost Modeling: Final Code with Feature Importance Plots
# =========================

# 1. Load Required Libraries
library(tidyverse)
library(caret)
library(randomForest)
library(xgboost)
library(e1071)
library(ROCR)
library(corrplot)
library(Matrix)
install.packages("vip")

library(vip)  # For feature importance plots in GLM
library(pROC) # For ROC curves

# 2. Load and Prepare Data
data <- read.csv("airbnb_ratings_new.csv", stringsAsFactors = FALSE)
data$host_is_superhost <- as.integer(data$Host.Is.Superhost)
data$Price <- as.numeric(gsub("[$,]", "", data$Price))

# Feature Engineering
data$reviews_per_day_available <- with(data, Reviews.per.month / (Availability.365 / 30 + 1))
review_vars <- c("Review.Scores.Rating", "Review.Scores.Cleanliness",
                 "Review.Scores.Checkin", "Review.Scores.Communication",
                 "Review.Scores.Location", "Review.Scores.Value")
data[review_vars] <- lapply(data[review_vars], function(x) as.numeric(as.character(x)))
data$avg_review_score <- rowMeans(data[, review_vars], na.rm = TRUE)

# Select top 6 variables based on previous correlation analysis
final_vars <- c("host_is_superhost", "Number.of.reviews", "Reviews.per.month",
                "Review.Scores.Cleanliness", "Review.Scores.Value",
                "avg_review_score", "reviews_per_day_available")
model_data <- data %>%
  dplyr::select(all_of(final_vars)) %>%
  na.omit()

# Standardize Features
scaled_features <- model_data %>%
  dplyr::select(-host_is_superhost) %>%
  scale() %>%
  as.data.frame()
model_data_scaled <- bind_cols(host_is_superhost = model_data$host_is_superhost, scaled_features)

# Split Data
set.seed(123)
trainIndex <- createDataPartition(model_data_scaled$host_is_superhost, p = 0.8, list = FALSE)
train <- model_data_scaled[trainIndex, ]
test  <- model_data_scaled[-trainIndex, ]

# ========================
# Logistic Regression
# ========================
model_log <- glm(host_is_superhost ~ ., data = train, family = "binomial")
pred_log <- predict(model_log, newdata = test, type = "response")
pred_class_log <- ifelse(pred_log > 0.5, 1, 0)
cm_log <- confusionMatrix(factor(pred_class_log), factor(test$host_is_superhost))
print("\nLogistic Regression Confusion Matrix")
print(cm_log)

# Precision, Recall, F1
res_log <- table(pred_class_log, test$host_is_superhost)
precision_log <- res_log[2,2]/sum(res_log[2,])
recall_log <- res_log[2,2]/sum(res_log[,2])
f1_log <- 2 * precision_log * recall_log / (precision_log + recall_log)
cat("\nLogistic - Precision:", precision_log, "Recall:", recall_log, "F1:", f1_log, "\n")

# Variable Importance Plot
vip::vip(model_log, num_features = 6, bar = TRUE, aesthetics = list(fill = "steelblue"),
         main = "Logistic Regression - Feature Importance")

# ========================
# Random Forest
# ========================
model_rf <- randomForest(factor(host_is_superhost) ~ ., data = train, ntree = 100, importance = TRUE)
pred_rf <- predict(model_rf, newdata = test)
cm_rf <- confusionMatrix(pred_rf, factor(test$host_is_superhost))
print("\nRandom Forest Confusion Matrix")
print(cm_rf)

res_rf <- table(pred_rf, test$host_is_superhost)
precision_rf <- res_rf[2,2]/sum(res_rf[2,])
recall_rf <- res_rf[2,2]/sum(res_rf[,2])
f1_rf <- 2 * precision_rf * recall_rf / (precision_rf + recall_rf)
cat("\nRandom Forest - Precision:", precision_rf, "Recall:", recall_rf, "F1:", f1_rf, "\n")

# Feature Importance Plot
varImpPlot(model_rf, main = "Random Forest - Feature Importance", pch = 16, col = "darkgreen")

# ========================
# XGBoost
# ========================
x_train <- model.matrix(host_is_superhost ~ . - 1, data = train)
x_test  <- model.matrix(host_is_superhost ~ . - 1, data = test)
y_train <- train$host_is_superhost
y_test  <- test$host_is_superhost
dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dtest  <- xgb.DMatrix(data = x_test,  label = y_test)

xgb_model <- xgboost(data = dtrain,
                     max.depth = 4,
                     nrounds = 100,
                     objective = "binary:logistic",
                     eval_metric = "error",
                     verbose = 0)

xgb_pred <- predict(xgb_model, dtest)
xgb_class <- ifelse(xgb_pred > 0.5, 1, 0)
cm_xgb <- confusionMatrix(factor(xgb_class), factor(y_test))
print("\nXGBoost Confusion Matrix")
print(cm_xgb)

res_xgb <- table(xgb_class, y_test)
precision_xgb <- res_xgb[2,2]/sum(res_xgb[2,])
recall_xgb <- res_xgb[2,2]/sum(res_xgb[,2])
f1_xgb <- 2 * precision_xgb * recall_xgb / (precision_xgb + recall_xgb)
cat("\nXGBoost - Precision:", precision_xgb, "Recall:", recall_xgb, "F1:", f1_xgb, "\n")

# XGBoost Feature Importance Plot
importance <- xgb.importance(feature_names = colnames(x_train), model = xgb_model)
xgb.plot.importance(importance, top_n = 6, main = "XGBoost - Feature Importance", rel_to_first = TRUE)


# ============================================
# SHAP VALUE EXPLANATION FOR XGBOOST MODEL
# Airbnb Superhost Prediction (Final Code)
# ============================================

# STEP 1: Install and Load Required SHAP Packages
# --------------------------------------------
# Only install once (uncomment if not installed)
# install.packages("shapviz")

library(shapviz)  # For SHAP value visualizations
library(xgboost)  # Already used for model training

# STEP 2: Create SHAP Object Using Trained XGBoost Model
# ------------------------------------------------------
# Ensure 'xgb_model' and 'x_train' are already created in your workflow
shap_values <- shapviz(xgb_model, X_pred = x_train)

# STEP 3: SHAP Visualization - Feature Importance
# -----------------------------------------------

# 3A: SHAP Bar Plot - Average Absolute Impact of Each Feature
sv_importance(shap_values, kind = "bar",
              main = "SHAP Feature Importance - Bar Plot")

# 3B: SHAP Beeswarm Plot - Instance-Level Feature Effects
sv_importance(shap_values, kind = "beeswarm",
              max_display = 15,                          # Top 15 features
              main = "SHAP Feature Importance - Beeswarm Plot")

# (Optional): You can also generate dependence plots per feature like this:
sv_dependence(shap_values, v = "avg_review_score")


# ========================================================
# PART 2: GMM CLUSTERING BASED ON SHAP-SELECTED FEATURES ---- SUBJECT TO CHANGE ACCORDING TO OTHER MODELS WE RUN BEFORE THIS
# ========================================================

# STEP 1: Load Required Libraries
library(mclust)
library(tidyverse)
library(ggplot2)
library(pheatmap)

# STEP 2: Select Top Features from SHAP for Clustering
gmm_vars <- c("Number.of.reviews", "Review.Scores.Rating", 
              "avg_review_score", "Reviews.per.month")

# STEP 3: Prepare & Clean Data
gmm_data <- data %>%
  dplyr::select(all_of(gmm_vars)) %>%
  mutate(
    log_reviews = log1p(Number.of.reviews),
    norm_avg_score = ifelse(avg_review_score > 30, NA, avg_review_score)
  ) %>%
  dplyr::select(log_reviews, norm_avg_score, Review.Scores.Rating, Reviews.per.month) %>%
  na.omit()

# STEP 4: Scale Features
gmm_scaled <- scale(gmm_data)

# STEP 5: Fit GMM Model
gmm_model <- Mclust(gmm_scaled)
gmm_data$Cluster <- as.factor(gmm_model$classification)

# STEP 6: Visualize GMM Clusters

# (A) Log Reviews vs. Avg Review Score
p1 <- ggplot(gmm_data, aes(x = log_reviews, y = norm_avg_score, color = Cluster)) +
  geom_point(alpha = 0.6, size = 2.5) +
  labs(title = "Refined GMM Clustering (Log Reviews vs. Avg Score)",
       x = "Log(Number of Reviews + 1)",
       y = "Normalized Avg Review Score") +
  theme_minimal() +
  scale_color_brewer(palette = "Set1")
print(p1)

# STEP 7: Cluster Profiling
profile_data <- gmm_data %>%
  bind_cols(
    data %>% 
      dplyr::select(host_is_superhost, Price, Accommodates, Bathrooms) %>% 
      dplyr::slice(1:nrow(gmm_data))
  )

cluster_summary <- profile_data %>%
  group_by(Cluster) %>%
  summarise(
    Count = n(),
    Superhost_Rate = round(mean(host_is_superhost, na.rm = TRUE), 3),
    Avg_Review_Score = round(mean(norm_avg_score), 2),
    Monthly_Reviews = round(mean(Reviews.per.month), 2),
    Avg_Rating = round(mean(Review.Scores.Rating), 2),
    Avg_Price = round(mean(Price, na.rm = TRUE), 2),
    Avg_Capacity = round(mean(Accommodates, na.rm = TRUE), 2),
    Avg_Bathrooms = round(mean(Bathrooms, na.rm = TRUE), 2)
  ) %>%
  arrange(desc(Superhost_Rate))
print(cluster_summary)

# STEP 8: Heatmap of Standardized Feature Profiles
features <- c("norm_avg_score", "Reviews.per.month", "Review.Scores.Rating",
              "Price", "Accommodates", "Bathrooms")

scaled_data <- profile_data %>%
  dplyr::select(all_of(features)) %>%
  scale() %>%
  as.data.frame()

scaled_data$Cluster <- profile_data$Cluster

z_score_summary <- scaled_data %>%
  group_by(Cluster) %>%
  summarise(across(everything(), mean)) %>%
  arrange(Cluster)

z_matrix <- z_score_summary %>%
  column_to_rownames("Cluster") %>%
  as.matrix()

pheatmap(z_matrix,
         cluster_rows = TRUE,
         cluster_cols = FALSE,
         color = colorRampPalette(c("red", "white", "blue"))(100),
         main = "Standardized Feature Profiles by Cluster",
         fontsize_row = 10,
         fontsize_col = 10)
###################
#k-nn
###################
#install.packages("FNN")
library(FNN)
library(caret)

# Ensure outcome is numeric (0/1) for FNN
y_train_num <- as.numeric(as.character(y_train))
y_test_num <- as.numeric(as.character(y_test))

k_values <- c(1, 3, 5, 7, 9)
results <- data.frame(k = integer(), Accuracy = double())

for (k in k_values) {
  set.seed(123)
  knn_pred <- knn(train = x_train, test = x_test, cl = y_train_num, k = k)
  pred_class <- as.factor(knn_pred)
  acc <- sum(pred_class == y_test) / length(y_test)
  results <- rbind(results, data.frame(k = k, Accuracy = acc))
}

print(results)
best_k <- results$k[which.max(results$Accuracy)]
cat("Best k selected:", best_k, "\n")

set.seed(123)
knn_pred <- knn(train = x_train, test = x_test, cl = y_train_num, k = best_k)

# Convert predictions to match y_test levels
knn_pred_class <- factor(as.character(as.numeric(knn_pred) - 1), levels = levels(y_test))
# Ensure y_test is a factor with proper levels
y_test <- factor(y_test, levels = c("0", "1"))

# Ensure predictions are also factors with the same levels
knn_pred_class <- factor(ifelse(knn_pred == "2", "1", "0"), levels = c("0", "1"))

# Now, check the levels again
levels(knn_pred_class)    # Should show "0" and "1"
levels(y_test)            # Should show "0" and "1"

# Evaluate
cm_knn <- confusionMatrix(knn_pred_class, y_test)
print(cm_knn)


# Extract precision, recall, F1
cm_table <- cm_knn$table
TP <- cm_table[2, 2]
TN <- cm_table[1, 1]
FP <- cm_table[2, 1]
FN <- cm_table[1, 2]

accuracy <- (TP + TN) / sum(cm_table)
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
f1 <- 2 * precision * recall / (precision + recall)

cat("K-NN Metrics:\n")
cat("Accuracy:", round(accuracy, 3), "\n")
cat("Precision:", round(precision, 3), "\n")
cat("Recall:", round(recall, 3), "\n")
cat("F1 Score:", round(f1, 3), "\n")

###################
#svm
###################
#library(e1071)

#svm_model <- svm(host_is_superhost ~ ., data = train, kernel = "radial", probability = TRUE)
#svm_model <- svm(host_is_superhost ~ ., data = train, kernel = "linear", probability = TRUE)
#svm_pred <- predict(svm_model, newdata = test)

#cm_svm <- confusionMatrix(svm_pred, factor(test$host_is_superhost))
#print(cm_svm)

# ========================================================
# PART 2: GMM CLUSTERING BASED ON SHAP-SELECTED FEATURES ---- SUBJECT TO CHANGE ACCORDING TO OTHER MODELS WE RUN BEFORE THIS
# ========================================================

# STEP 1: Load Required Libraries
library(mclust)
library(tidyverse)
library(ggplot2)
library(pheatmap)

# STEP 2: Select Top SHAP Features for Clustering
gmm_vars <- c("Number.of.reviews", "Reviews.per.month",
              "Review.Scores.Rating", "avg_review_score")

# STEP 3: Prepare & Clean Data
gmm_data <- data %>%
  select(all_of(gmm_vars)) %>%
  mutate(
    log_reviews = log1p(Number.of.reviews),                        # log-scale to reduce skew
    norm_avg_score = ifelse(avg_review_score > 30, NA, avg_review_score)  # remove outliers
  ) %>%
  select(log_reviews, norm_avg_score, Review.Scores.Rating, Reviews.per.month) %>%
  na.omit()  # remove rows with NAs in selected vars


# STEP 4: Scale Features
gmm_scaled <- scale(gmm_data)

# STEP 5: Fit GMM Model
gmm_model <- Mclust(gmm_scaled)
gmm_data$Cluster <- as.factor(gmm_model$classification)

# STEP 6: Visualize GMM Clusters
p1 <- ggplot(gmm_data, aes(x = log_reviews, y = norm_avg_score, color = Cluster)) +
  geom_point(alpha = 0.6, size = 2.5) +
  labs(title = "GMM Clustering: Log(Reviews) vs. Avg Score",
       x = "Log(Number of Reviews + 1)",
       y = "Normalized Avg Review Score") +
  theme_minimal() +
  scale_color_brewer(palette = "Set1")
print(p1)

# STEP 7: Cluster Profiling
profile_data <- gmm_data %>%
  bind_cols(
    data %>% 
      select(host_is_superhost, Price, Accommodates, Bathrooms) %>%
      dplyr::slice(1:nrow(gmm_data))  # Ensure row alignment
  )

cluster_summary <- profile_data %>%
  group_by(Cluster) %>%
  summarise(
    Count = n(),
    Superhost_Rate = round(mean(host_is_superhost, na.rm = TRUE), 3),
    Avg_Review_Score = round(mean(norm_avg_score), 2),
    Monthly_Reviews = round(mean(Reviews.per.month), 2),
    Avg_Rating = round(mean(Review.Scores.Rating), 2),
    Avg_Price = round(mean(Price, na.rm = TRUE), 2),
    Avg_Capacity = round(mean(Accommodates, na.rm = TRUE), 2),
    Avg_Bathrooms = round(mean(Bathrooms, na.rm = TRUE), 2)
  ) %>%
  arrange(desc(Superhost_Rate))
print(cluster_summary)

# STEP 8: Heatmap of Standardized Cluster Profiles
features <- c("norm_avg_score", "Reviews.per.month", "Review.Scores.Rating",
              "Price", "Accommodates", "Bathrooms")

scaled_data <- profile_data %>%
  select(all_of(features)) %>%
  scale() %>%
  as.data.frame()

scaled_data$Cluster <- profile_data$Cluster

z_score_summary <- scaled_data %>%
  group_by(Cluster) %>%
  summarise(across(everything(), mean)) %>%
  arrange(Cluster)

z_matrix <- z_score_summary %>%
  column_to_rownames("Cluster") %>%
  as.matrix()

pheatmap(z_matrix,
         cluster_rows = TRUE,
         cluster_cols = FALSE,
         color = colorRampPalette(c("red", "white", "blue"))(100),
         main = "Standardized Feature Profiles by Cluster",
         fontsize_row = 10,
         fontsize_col = 10)



