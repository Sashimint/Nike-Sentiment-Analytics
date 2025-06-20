# Nike Consumer Sentiment Analytics 📉👟

This is a university project built with R for the course **BC2407: Analytics II – Advanced Predictive Techniques** at Nanyang Technological University (NTU). It explores consumer dissatisfaction patterns toward Nike using both transactional data and customer reviews.

---

## 📌 Objective

To identify drivers of dissatisfaction and segment behavior across two datasets:
- **SiteJabber Reviews**: Text, ratings, and sentiment features
- **Nike Purchase History**: Transactional data with discounts, returns, and membership

---

## 📁 Repository Structure
R-Analytics-Project-NikeSentiment/
├── analysis/ # All R scripts (review + purchase analysis)
│ ├── purchase_analysis_models.R
│ └── review_sentiment_models.R
├── data/ # Cleaned CSVs (anonymized)
│ ├── nike_purchases_cleaned.csv
│ └── nike_reviews_cleaned.csv
├── visuals/ # Output plots organized by type
│ ├── eda/
│ ├── pdp/
│ ├── clustering/
│ ├── sentiment/
│ └── mars_model/
│ └── association_rules/
│ └── quantile_regression/
├── .gitignore # Prevents tracking of dev/backup files
└── README.md # This file
