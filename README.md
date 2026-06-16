
# Task 6 - Data Transformation & Feature Engineering

## Tools Used
Python, Pandas, Scikit-learn

## Dataset
Housing Prices Dataset - Kaggle
Link: https://www.kaggle.com/datasets/yasserh/housing-prices-dataset

## Steps Performed
1. One-hot encoding on categorical columns
2. Standard Scaling on numeric columns
3. Created interaction term: area x bedrooms

## Interview Answers
**Scaling vs Normalization?**
Scaling (StandardScaler) converts data to mean=0, std=1.
Normalization converts data to range 0-1. Scaling is used when data has outliers.

**Why use One-hot encoding?**
ML models cannot understand text/categorical data. One-hot converts categories into 0/1 numeric columns so model can process them.
