#===============================================================================
# Dataset 2: purchase_history_cleaned.csv
#===============================================================================

# Uncomment and run the line below if you need to install packages
# install.packages(c("data.table", "dplyr", "ggplot2", "earth", "randomForest", 
#                    "caret", "pdp", "factoextra", "skimr","DataExplorer"))

# -----------------------------------
# Load Libraries
# -----------------------------------
library(tidyverse)
library(data.table)
library(skimr)
library(DataExplorer)
library(corrplot)
library(earth)
library(randomForest)
library(caret)
library(pdp)
library(factoextra)

# -----------------------------------
# Clean data
# -----------------------------------
nike_df <- fread("purchase_history_cleaned.csv")

# View structure and summary
str(nike_df)
summary
skim(nike_df)
colSums(is.na(nike_df))
plot_missing(nike_df)

# Encode discount type as numeric factor
nike_df$nike_disctype <- as.integer(factor(nike_df$nike_disctype))

# Replace missing values with 0
nike_df[is.na(nike_df)] <- 0

# Create membership flag (for MARS visuals)
nike_df$membership <- ifelse(nike_df$member_id == "0", "Non-Member", "Member")

# -----------------------------------
# EDA Visualisations
# -----------------------------------

# Histograms for numeric variables
nike_df %>%
  select(where(is.numeric), -nike_disctype) %>%   # exclude variable
  gather() %>%
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram(bins = 30, fill = "skyblue", color = "black") +
  labs(title = "Distribution of Numeric Variables")

# Bar plots for categorical variables 
nike_df %>%
  select(where(is.character), -nike_disctype) %>%   # exclude variable
  gather(key = "variable", value = "value") %>%
  ggplot(aes(x = value)) +
  facet_wrap(~ variable, scales = "free") +
  geom_bar(fill = "orange") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Distribution of Categorical Variables")

# Correlation heatmap 
num_data <- nike_df %>% select(where(is.numeric), -nike_disctype)
cor_matrix <- cor(num_data, use = "complete.obs")
corrplot(cor_matrix, method = "circle", type = "upper")

# Membership vs Returns
avg_returns <- nike_df %>%
  group_by(membership) %>%
  summarise(avg_returns = mean(nike_returns, na.rm = TRUE))

ggplot(avg_returns, aes(x = membership, y = avg_returns, fill = membership)) +
  geom_bar(stat = "identity", width = 0.5) +
  labs(title = "Average Nike Returns by Membership Status",
       x = "Membership Status", y = "Avg Number of Returns") +
  theme_minimal() +
  theme(legend.position = "none", plot.title = element_text(hjust = 0.5))

# -----------------------------------
# Random Forest Models
# -----------------------------------

# Model 1: Predict Shoes Bought
rf_model1 <- randomForest(nike_shoes ~ nike_discount + nike_returns + nike_disctype, data = nike_df)
varImpPlot(rf_model1, main = "Variable Importance for Shoes Bought")

# Model 2: Predict Returns
rf_model2 <- randomForest(nike_returns ~ nike_discount + nike_shoes + nike_disctype, data = nike_df)

# PDP: Shoes Bought vs Discount
pdp_shoes <- partial(rf_model1, pred.var = "nike_discount", plot = FALSE)
ggplot(pdp_shoes, aes(x = nike_discount, y = yhat)) +
  geom_line(color = "blue") +
  labs(title = "PDP: Shoes Bought vs Discount",
       x = "Discount (%)", y = "Predicted Shoes Bought") +
  theme_minimal()

# PDP: Returns vs Discount
pdp_returns <- partial(rf_model2, pred.var = "nike_discount", plot = FALSE)
ggplot(pdp_returns, aes(x = nike_discount, y = yhat)) +
  geom_line(color = "red") +
  labs(title = "PDP: Returns vs Discount",
       x = "Discount (%)", y = "Predicted Returns") +
  theme_minimal()

# Visualise Shoes Bought vs Returns
ggplot(nike_df, aes(x = nike_shoes, y = nike_returns, color = factor(nike_disctype))) +
  geom_point(size = 3) +
  labs(title = "Shoes Bought vs Returned", color = "Discount Type") +
  theme_minimal()

# -----------------------------------
# MARS Model
# -----------------------------------

mars_model <- earth(nike_returns ~ nike_discount + nike_shoes, degree = 1, data = nike_df)

summary(mars_model)

# Evaluate performance
mars_pred <- predict(mars_model)
mars_rmse <- round(sqrt(mean((nike_df$nike_returns - mars_pred)^2)), 3)
cat("MARS RMSE:", mars_rmse, "\n")

# -----------------------------------
# 6. Clustering Analysis
# -----------------------------------

# Scale relevant columns
scaled_df <- scale(nike_df[, .(nike_shoes, nike_discount, nike_returns, nike_disctype)])

# Find optimal clusters
fviz_nbclust(scaled_df, kmeans, method = "wss")

# Apply K-Means clustering (k = 4)
set.seed(123)
kmeans_out <- kmeans(scaled_df, centers = 4, nstart = 25)
nike_df$cluster <- kmeans_out$cluster

# Visualize clusters
fviz_cluster(list(data = scaled_df, cluster = kmeans_out$cluster),
             main = "K-Means Clustering Result")

# Cluster vs Discount Type
table(nike_df$cluster, nike_df$nike_disctype)

# Return Rates by Cluster (k = 2 for comparison)
return_data <- nike_df[, .(nike_returns)]
scaled_returns <- scale(return_data)
fviz_nbclust(scaled_returns, kmeans, method = "wss", k.max = 3)

set.seed(123)
kmeans_return <- kmeans(scaled_returns, centers = 2, nstart = 25)
nike_df$return_cluster <- kmeans_return$cluster

# Plot Return Distribution by Cluster
ggplot(nike_df, aes(x = nike_returns, fill = factor(return_cluster))) +
  geom_histogram(binwidth = 0.2, position = "dodge", color = "black") +
  labs(title = "Return Rates Distribution by Cluster",
       x = "Returns", y = "Frequency", fill = "Cluster") +
  theme_minimal()

# Summary by Cluster
cluster_summary <- nike_df %>%
  group_by(return_cluster) %>%
  summarise(avg_returns = mean(nike_returns, na.rm = TRUE),
            total_members = n())

print(cluster_summary)

#===============================================================================
# END OF R SCRIPT
#===============================================================================