# 🚗 Used Car Price Prediction — Rusty Bargain

Predicting used car market value using gradient boosting models, with a focus on prediction quality, prediction speed, and training time.

---

## 📌 Overview

Rusty Bargain, a used car sales service, needs a model to quickly estimate the market value of a car based on historical data — technical specs, trim levels, and prices.

**Business priorities:**
- 🎯 Prediction quality (RMSE)
- ⚡ Prediction speed
- ⏱️ Training time

---

## 📊 Dataset

| Feature | Description |
|---|---|
| `Price` | Vehicle price (target) |
| `VehicleType` | Body type |
| `RegistrationYear` | Year of first registration |
| `Gearbox` | Transmission type |
| `Power` | Engine power (hp) |
| `Model` | Car model |
| `Mileage` | Kilometers driven |
| `RegistrationMonth` | Month of first registration |
| `FuelType` | Fuel type |
| `Brand` | Car brand |
| `NotRepaired` | Whether the car has unrepaired damage |

- **Raw records:** 354,369
- **After cleaning:** outliers and invalid entries removed

---

## 🔧 Methodology

### 1. Data Cleaning
- Removed prices ≤ 100 (invalid listings)
- Removed invalid registration years (< 1900 or > 2016)
- Removed vehicles with power = 0 or > 1,000 hp
- Dropped uninformative columns: `NumberOfPictures`, `DateCrawled`, `DateCreated`, `LastSeen`
- Filled missing categorical values with `"unknown"` (absence of info is informative)
- Replaced `RegistrationMonth = 0` with column median

### 2. Feature Engineering
- One-Hot Encoding for tree-based and linear models
- CatBoost received raw categorical features natively

### 3. Train / Validation / Test Split
- 60% train / 20% validation / 20% test (stratified random split)

### 4. Models Trained
- Linear Regression (baseline)
- Decision Tree
- Random Forest
- LightGBM
- CatBoost

---

## 📈 Results

| Model | RMSE (Validation) | Train Time | Predict Time |
|---|---|---|---|
| **LightGBM** | **1,665** ✅ | 2.5s | 0.30s |
| CatBoost | 1,678 | 28.6s | 0.09s |
| Random Forest | 1,884 | 40.4s | 0.19s |
| Decision Tree | 1,983 | 2.5s | 0.07s |
| Linear Regression | 2,612 | 6.4s | 0.10s |

**Final test RMSE (LightGBM): 1,654** — well below the 2,500 threshold.

---

## 🏆 Best Model: LightGBM

LightGBM was selected as the final model for offering the best balance between:
- Lowest RMSE on both validation and test sets
- Fast training time (2.5s vs. 40s for Random Forest)
- Strong generalization with no overfitting

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-lightgrey)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-Boosting-green)
![CatBoost](https://img.shields.io/badge/CatBoost-Boosting-yellow)

- **Python** — Pandas, NumPy
- **Machine Learning** — Scikit-learn, LightGBM, CatBoost
- **Visualization** — Seaborn, Matplotlib

---

## 📁 Project Structure

```
numerical-methods/
│
├── sprint13.ipynb       # Full analysis and modeling
├── README.md
└── .gitignore
```

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/laaurasaporski/numerical-methods.git

# Install dependencies
pip install pandas numpy scikit-learn lightgbm catboost seaborn matplotlib

# Open the notebook
jupyter notebook sprint13.ipynb
```

---

## 💡 Key Insights

- **Gradient boosting models significantly outperform** linear and tree-based baselines for this task
- **LightGBM is the top choice** when both accuracy and speed matter — 16x faster to train than Random Forest with better RMSE
- **CatBoost handles raw categoricals natively**, eliminating the need for OHE, but at higher training cost
- **Missing values in categorical features were preserved** as `"unknown"` rather than dropped, contributing to model robustness
