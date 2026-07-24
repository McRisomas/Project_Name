# 🏠 Ames Housing Price Prediction

Predicting house sale prices using the Ames Housing dataset 
through a complete end-to-end machine learning pipeline.

---

## 📋 Project Overview

This project demonstrates a full ML pipeline including:
- Exploratory Data Analysis (EDA) and Data Cleaning
- Feature engineering (composite scores, interaction terms,
  temporal features, mean encoding, domain knowledge)
- Feature selection (VIF analysis, redundancy checks)
- Regularized regression modeling (Ridge, Lasso)

**Final RMSE:** 0.1052  
**Kaggle Score:** 0.12711

---

## 🔍 Key Findings

- Created `qual_x_size` interaction feature (quality × total area)
  which showed strong correlation with SalePrice
- `neigh_mean` (target encoding of Neighborhood) was among the
  most impactful features
- Log-transforming skewed features (LotArea, TotalBsmtSF, etc.)
  significantly improved model performance
- Identified and handled multicollinearity between engineered
  features using VIF analysis

---

## 🛠️ Pipeline Summary

Raw Data
↓ Data Cleaning (missing values, outliers)
↓ Feature Engineering (15+ new features)
↓ Feature Selection (VIF + domain knowledge)
↓ Encoding (ordinal + one-hot + target encoding)
↓ Log Transform (skewed features)
↓ Modeling (Ridge, Lasso, comparison)
↓ Evaluation (5-fold CV, RMSE)

## 📊 Results

| Model         | CV RMSE  | Std RMSE | 
|---------------|----------|----------|
| Lasso         | 0.111610 | 0.004193 |
| ElasticNet    | 0.111681 | 0.004168 |
| Ridge         | 0.112891 | 0.004327 |
| Linear        | 0.115959 | 0.004518 |
| Random Forest | 0.126437 | 0.006049 |

---

## 🗂️ Repository Structure

├── notebooks/
│ ├── HousePricing.ipynb
├── data/
| ├── processed/
| ├── raw/
├── requirements.txt
└── README.md

## ⚙️ How to Run

```bash
git clone https://github.com/McRisomas/Ames_Housing_Price_Prediction
cd Ames_Housing_Price_Prediction
pip install -r requirements.txt
jupyter notebook
```

---

## 📚 References

- Dataset: [Ames Housing Dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- Book: Aurélien Géron — *Hands-On Machine Learning* (Ch. 4)
