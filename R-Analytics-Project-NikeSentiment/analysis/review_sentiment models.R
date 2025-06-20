#===============================================================================
# Dataset 1: nike_reviews_cleaned.csv
#===============================================================================


# Uncomment and run the line below if you need to install packages
# install.packages(c("tidyverse", "lubridate", "ggplot2", "corrplot", "tidytext", "tm", "SnowballC", "wordcloud", 
# "textdata", "randomForest", "caret", "quantreg", "arules", "arulesViz", "earth", "plotmo"))

# Load required libraries
library(tidyverse)
library(lubridate)
library(ggplot2)
library(corrplot)
library(tidytext)
library(tm)
library(SnowballC)
library(wordcloud)
library(textdata)
library(randomForest)
library(caret)
library(quantreg)
library(arules)
library(arulesViz)
library(earth)
library(plotmo)

# Load stop words for text cleaning
data("stop_words")

#---------------------------------
# Load and clean data
#---------------------------------

df <- read.csv("nike_reviews_cleaned.csv")

# inspect structure and missing values
str(df)
summary(df)
colSums(is.na(df))


# convert 'date_updated' to proper Date format
df$date_updated <- as.Date(df$date_updated, format = "%d/%m/%Y")
sum(is.na(df$date_updated))
df <- df %>% filter(!is.na(date_updated))


# convert relevant rating columns to numeric
cat_rating_cols <- c("service_rating", "value_rating", "shipping_rating", 
                     "returns_rating", "quality_rating", "overall_rating")
df[cat_rating_cols] <- lapply(df[cat_rating_cols], function(x) as.numeric(as.character(x)))


# convert invalid 0s to NA, then impute using overall_rating
df[cat_rating_cols[-length(cat_rating_cols)]] <- lapply(df[cat_rating_cols[-length(cat_rating_cols)]], function(x) ifelse(x == 0, NA, x))
for (col in cat_rating_cols[-length(cat_rating_cols)]) {
  df[[col]] <- ifelse(is.na(df[[col]]), df$overall_rating, df[[col]])
}

# create behavioral flags
df$helpful_review <- ifelse(df$vote_count > 0, "Yes", "No")
df$trust_gap <- ifelse(df$quality_rating >= 4 & df$service_rating <= 2, "Yes", "No")
df$frustrated_buyer <- ifelse(
  ((df$quality_rating >= 3 | df$value_rating >= 3) &
     (df$service_rating <= 2 | df$shipping_rating <= 2 | df$returns_rating <= 2)) |
    (df$overall_rating == 1),
  "Yes", "No"
)

#---------------------------------
# EDA & Visualisations
#---------------------------------

# Histogram of reviews over time
ggplot(df, aes(x = date_updated)) +
  geom_histogram(binwidth = 30, fill = "steelblue", color = "white") +
  labs(title = "Review Volume Over Time", x = "Date", y = "Count")

# Overall rating distribution
ggplot(df, aes(x = factor(overall_rating))) +
  geom_bar(fill = "tomato") +
  labs(title = "Distribution of Overall Ratings", x = "Rating", y = "Count")

# Correlation heatmap of rating dimensions
ratings <- df %>%
  select(overall_rating, service_rating, value_rating,
         shipping_rating, returns_rating, quality_rating) %>%
  drop_na()

corrplot(cor(ratings), method = "color", type = "upper", addCoef.col = "black")

# Vote count by rating – are low/high ratings more “helpful”?
ggplot(df, aes(x = factor(overall_rating), y = vote_count)) +
  geom_boxplot(fill = "lightblue") +
  labs(title = "Helpful Vote Count by Overall Rating", x = "Overall Rating", y = "Vote Count")

#---------------------------------
# Text Mining & Sentiment Analysis
#---------------------------------

# Add review ID for tracking
df <- df %>%
  mutate(review_id = row_number())

#Tokenize and clean review text
review_words <- df %>%
  filter(!is.na(review_content)) %>%
  select(review_id, review_content) %>%
  unnest_tokens(word, review_content) %>%
  anti_join(stop_words, by = "word")

# Top 20 most common words
top_words <- review_words %>%
  count(word, sort = TRUE) %>%
  slice_max(n, n = 20)

ggplot(top_words, aes(x = reorder(word, n), y = n)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 20 Most Common Words in Reviews", x = "Word", y = "Frequency") +
  theme_minimal()

# Word cloud
set.seed(123)
word_freq <- review_words %>%
  count(word, sort = TRUE)

wordcloud(words = word_freq$word, freq = word_freq$n,
          min.freq = 5, max.words = 100,
          colors = brewer.pal(8, "Dark2"),
          random.order = FALSE)

# Sentiment classification using Bing lexicon
sentiment_counts <- review_words %>%
  inner_join(get_sentiments("bing"), by = "word") %>%
  count(sentiment) %>%
  mutate(sentiment = str_to_title(sentiment))

ggplot(sentiment_counts, aes(x = sentiment, y = n, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  labs(title = "Sentiment of Review Words", x = "Sentiment", y = "Count") +
  scale_fill_manual(values = c("Positive" = "seagreen", "Negative" = "firebrick")) +
  theme_minimal()

# Top sentiment words
top_sentiment_words <- review_words %>%
  inner_join(get_sentiments("bing"), by = "word") %>%
  count(word, sentiment, sort = TRUE) %>%
  group_by(sentiment) %>%
  slice_max(n, n = 10) %>%
  ungroup()

ggplot(top_sentiment_words, aes(x = reorder(word, n), y = n, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~sentiment, scales = "free") +
  coord_flip() +
  labs(title = "Top Words Contributing to Sentiment", x = "Word", y = "Frequency") +
  scale_fill_manual(values = c("positive" = "darkgreen", "negative" = "darkred")) +
  theme_minimal()

# Compute sentiment score per review
review_sentiment <- review_words %>%
  inner_join(get_sentiments("bing"), by = "word") %>%
  count(review_id, sentiment) %>%
  pivot_wider(names_from = sentiment, values_from = n, values_fill = 0) %>%
  mutate(sentiment_score = positive - negative)

# Merge back into main dataframe
df <- df %>%
  left_join(review_sentiment, by = "review_id") %>%
  mutate(sentiment_score = replace_na(sentiment_score, 0))  # assign 0 to reviews with no matched words

# Plot: Distribution of sentiment scores
ggplot(df, aes(x = sentiment_score)) +
  geom_histogram(binwidth = 1, fill = "mediumpurple", color = "white") +
  labs(title = "Sentiment Score Distribution", x = "Sentiment Score", y = "Number of Reviews") +
  theme_minimal()

# Plot: Sentiment score vs overall rating
ggplot(df, aes(x = factor(overall_rating), y = sentiment_score)) +
  geom_boxplot(fill = "skyblue") +
  labs(title = "Sentiment Score by Overall Rating",
       x = "Overall Rating", y = "Sentiment Score") +
  theme_minimal()

# Optional: Correlation coefficient
cor(df$overall_rating, df$sentiment_score, use = "complete.obs")

# -----------------------------
# association rules: recent reviews
# -----------------------------

# split data into recent and older based on review date
recent_df <- subset(df, date_updated >= as.Date("2022-01-01"))
older_df  <- subset(df, date_updated < as.Date("2022-01-01"))

# quick checks
table(df$helpful_review)
table(df$trust_gap)
table(df$frustrated_buyer)

arules_df <- recent_df %>%
  select(service_rating, value_rating, shipping_rating, returns_rating, 
         quality_rating, overall_rating, helpful_review, trust_gap, frustrated_buyer)
arules_df[] <- lapply(arules_df, as.factor)
trans <- as(arules_df, "transactions")

# generate association rules for frustrated buyers
rules_frustrated <- apriori(trans,
                            parameter = list(supp = 0.02, conf = 0.4, minlen = 2),
                            appearance = list(rhs = "frustrated_buyer=Yes", default = "lhs"))
cat("\n--- frustrated buyer rules (recent reviews) ---\n")
if (length(rules_frustrated) > 0) {
  inspect(sort(rules_frustrated, by = "lift")[1:min(10, length(rules_frustrated))])
} else {
  cat("no rules found.\n")
}

# generate association rules for trust gap
rules_trustgap <- apriori(trans,
                          parameter = list(supp = 0.02, conf = 0.4, minlen = 2),
                          appearance = list(rhs = "trust_gap=Yes", default = "lhs"))
cat("\n--- trust gap rules (recent reviews) ---\n")
if (length(rules_trustgap) > 0) {
  inspect(sort(rules_trustgap, by = "lift")[1:min(10, length(rules_trustgap))])
} else {
  cat("no rules found.\n")
}

# generate rules for helpful reviews with low scores
rules_helpful_low <- apriori(trans,
                             parameter = list(supp = 0.02, conf = 0.4, minlen = 2),
                             appearance = list(rhs = c("overall_rating=1", "helpful_review=Yes"), default = "lhs"))
cat("\n--- helpful low score rules (recent reviews) ---\n")
if (length(rules_helpful_low) > 0) {
  inspect(sort(rules_helpful_low, by = "lift")[1:min(10, length(rules_helpful_low))])
} else {
  cat("no rules found.\n")
}

# -----------------------------
# association rules: older reviews
# -----------------------------
older_df$helpful_review <- ifelse(older_df$vote_count > 0, "Yes", "No")
older_df$trust_gap <- ifelse(older_df$quality_rating >= 4 & older_df$service_rating <= 2, "Yes", "No")
older_df$frustrated_buyer <- ifelse(
  ((older_df$quality_rating >= 3 | older_df$value_rating >= 3) &
     (older_df$service_rating <= 2 | older_df$shipping_rating <= 2 | older_df$returns_rating <= 2)) |
    (older_df$overall_rating == 1),
  "Yes", "No"
)

arules_old <- older_df %>%
  select(service_rating, value_rating, shipping_rating, returns_rating, 
         quality_rating, overall_rating, helpful_review, trust_gap, frustrated_buyer)
arules_old[] <- lapply(arules_old, as.factor)
trans_old <- as(arules_old, "transactions")

# set thresholds
safe_support <- 0.02
safe_conf <- 0.4

# frustrated buyer rules
rules_frustrated_old <- apriori(trans_old,
                                parameter = list(supp = safe_support, conf = safe_conf, minlen = 2),
                                appearance = list(rhs = "frustrated_buyer=Yes", default = "lhs"))
cat("\n--- frustrated buyer rules (older reviews) ---\n")
if (length(rules_frustrated_old) > 0) {
  inspect(sort(rules_frustrated_old, by = "lift")[1:min(10, length(rules_frustrated_old))])
} else {
  cat("no rules found.\n")
}

# trust gap rules
rules_trust_old <- apriori(trans_old,
                           parameter = list(supp = safe_support, conf = safe_conf, minlen = 2),
                           appearance = list(rhs = "trust_gap=Yes", default = "lhs"))
cat("\n--- trust gap rules (older reviews) ---\n")
if (length(rules_trust_old) > 0) {
  inspect(sort(rules_trust_old, by = "lift")[1:min(10, length(rules_trust_old))])
} else {
  cat("no rules found.\n")
}

# helpful low score rules
rules_helpful_low_old <- apriori(trans_old,
                                 parameter = list(supp = safe_support, conf = safe_conf, minlen = 2),
                                 appearance = list(rhs = c("overall_rating=1", "helpful_review=Yes"), default = "lhs"))
cat("\n--- helpful low score rules (older reviews) ---\n")
if (length(rules_helpful_low_old) > 0) {
  inspect(sort(rules_helpful_low_old, by = "lift")[1:min(10, length(rules_helpful_low_old))])
} else {
  cat("no rules found.\n")
}


# --- Helper function to safely plot top rules ---
plot_top_rules <- function(rules, title) {
  if (length(rules) > 0) {
    top <- sort(rules, by = "lift")[1:min(10, length(rules))]
    cat(paste0("\nVisualizing: ", title, "\n"))
    plot(top, method = "graph", engine = "htmlwidget", main = title)
  } else {
    cat(paste0("\nNo rules found for: ", title, "\n"))
  }
}

# === RECENT DATASET VISUALS ===
plot_top_rules(rules_frustrated, "Recent Reviews – Frustrated Buyer Rules")
plot_top_rules(rules_trustgap, "Recent Reviews – Trust Gap Rules")
plot_top_rules(rules_helpful_low, "Recent Reviews – Helpful + Low Score Rules")

# === OLDER DATASET VISUALS ===
plot_top_rules(rules_frustrated_old, "Older Reviews – Frustrated Buyer Rules")
plot_top_rules(rules_trust_old, "Older Reviews – Trust Gap Rules")
plot_top_rules(rules_helpful_low_old, "Older Reviews – Helpful + Low Score Rules")

# -------------------------
# MARS MODEL
# -------------------------

# defining the formula
mars_formula <- overall_rating ~ service_rating + value_rating + shipping_rating +
  returns_rating + quality_rating +
  trust_gap + frustrated_buyer + helpful_review

# split data by date
train_df <- df[df$date_updated < as.Date("2022-01-01"), ]
test_df  <- df[df$date_updated >= as.Date("2022-01-01"), ]

# ensure that there are no NA values in predictors
colSums(is.na(train_df[, all.vars(mars_formula)]))

# run the model
mars_model <- earth(mars_formula, data = train_df)

# variable importance plot
evimp(mars_model)
plot(evimp(mars_model))

# predict on Test Set
preds <- predict(mars_model, newdata = test_df)
actual <- test_df$overall_rating

# evaluate the model performance
rmse_val <- RMSE(preds, actual)
mae_val <- MAE(preds, actual)

cat("Test RMSE:", round(rmse_val, 3), "\n")
cat("Test MAE :", round(mae_val, 3), "\n")

# -------------------------
# QUANTILE REGRESSION MODEL
# -------------------------

# Prepare data for QR model
qr_data <- df %>%
  select(overall_rating, sentiment_score, vote_count,
         service_rating, value_rating, quality_rating) %>%
  drop_na()

# Fit quantile regression models at τ = 0.25, 0.5 (median), 0.75
qr_25 <- rq(overall_rating ~ sentiment_score + vote_count + service_rating +
              value_rating + quality_rating,
            tau = 0.25, data = qr_data)

qr_50 <- rq(overall_rating ~ sentiment_score + vote_count + service_rating +
              value_rating + quality_rating,
            tau = 0.5, data = qr_data)

qr_75 <- rq(overall_rating ~ sentiment_score + vote_count + service_rating +
              value_rating + quality_rating,
            tau = 0.75, data = qr_data)

# View summaries
summary(qr_25)
summary(qr_50)
summary(qr_75)

# Plot coefficient paths across deciles
taus <- seq(0.1, 0.9, by = 0.1)
qr_fit <- rq(overall_rating ~ sentiment_score + vote_count + service_rating +
               value_rating + quality_rating,
             tau = taus, data = qr_data)

plot(summary(qr_fit), main = "Quantile Regression Coefficient Paths")

#===============================================================================
# END OF R SCRIPT
#===============================================================================