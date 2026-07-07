# credit-score-classification-

Automated credit score classification using LightGBM with auto-regressive sequential prediction.

## 🚀 Project Overview

This project classifies client creditworthiness into **Good**, **Standard**, and **Poor** brackets. It implements:

1. **Dynamic Customer Profiles** (aggregations) and sequential lag/difference feature engineering.
2. **Linear Credit History Age Reconstruction** to correct structural timeline progression.
3. **Auto-Regressive Forward Predicting Loop** to simulate real-world sequence predictions on future held-out months (September–December) from previous predictions.
4. **Interactive Dark-Themed Streamlit Dashboard** containing exploratory dashboards, model assessment, and a real-time predictor playground.

## 📊 Performance

- **Accuracy:** 83.1%
- **Macro F1 Score:** 0.831

## 🛠️ Getting Started

```bash
# 1. Install dependencies
pip install pandas numpy lightgbm scikit-learn streamlit matplotlib seaborn joblib

# 2. Train the model
python train.py

# 3. Run sequential inference on the test set
python predict.py

# 4. Start the interactive dashboard
streamlit run dashboard.py
```
