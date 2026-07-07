import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import joblib

NUMERIC_FEATURES = [
    'Age_clean', 'Annual_Income_clean', 'Monthly_Inhand_Salary_clean',
    'Num_Bank_Accounts_clean', 'Num_Credit_Card_clean', 'Interest_Rate_clean',
    'Num_of_Loan_clean', 'Delay_from_due_date_clean', 'Num_of_Delayed_Payment_clean',
    'Changed_Credit_Limit_clean', 'Num_Credit_Inquiries_clean', 'Outstanding_Debt_clean',
    'Total_EMI_per_month', 'Amount_invested_monthly_clean', 'Monthly_Balance_clean',
    'History_Age_Months_Imputed'
]

LOAN_TYPES = [
    'Payday_Loan', 'Credit_Builder_Loan', 'Not_Specified', 'Home_Equity_Loan',
    'Student_Loan', 'Mortgage_Loan', 'Personal_Loan', 'Debt_Consolidation_Loan', 'Auto_Loan'
]

CAT_COLS = ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour']

def get_features_list(profiles_cols, include_target_lag=True):
    """
    Returns the comprehensive list of features to be fed into the model.
    """
    features = NUMERIC_FEATURES.copy()
    
    # Profile columns
    features.extend(profiles_cols)
    
    # Lags columns of dynamic metrics
    lag_cols = [
        'Outstanding_Debt_clean', 'Delay_from_due_date_clean', 'Num_of_Delayed_Payment_clean',
        'Changed_Credit_Limit_clean', 'Num_Credit_Inquiries_clean', 'Monthly_Balance_clean',
        'Amount_invested_monthly_clean', 'Total_EMI_per_month'
    ]
    for col in lag_cols:
        features.extend([f"{col}_lag1", f"{col}_diff1", f"{col}_lag2", f"{col}_diff2"])
        
    # Loan indicators
    loan_features = ['Loan_' + loan for loan in LOAN_TYPES]
    features.extend(loan_features)
    
    # Categoricals
    features.extend(CAT_COLS)
    
    if include_target_lag:
        features.append('target_lag1')
        
    return features

def train_model(X_train, y_train, cat_features, num_boost_round=230):
    """
    Trains a LightGBM model with optimized hyperparameters.
    """
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
    
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 127,
        'max_depth': -1,
        'feature_fraction': 0.8,
        'verbose': -1,
        'random_state': 42
    }
    
    print(f"Training LightGBM model for {num_boost_round} rounds...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round
    )
    return model

def sequential_predict(model, df_to_predict, start_month=9, end_month=12, features=None, categorical_features=None):
    """
    Performs month-by-month sequential predictions, auto-regressively updating the target_lag1 feature.
    Assumes df_to_predict has been preprocessed and sorted.
    """
    df = df_to_predict.copy()
    df = df.sort_values(['Customer_ID', 'Month_Val']).reset_index(drop=True)
    
    # Initialize empty column for predictions
    df['pred'] = np.nan
    
    for month in range(start_month, end_month + 1):
        print(f"Predicting for Month {month}...")
        
        # 1. Update target_lag1 from previous month's predictions (if month > start_month)
        if month > start_month:
            prev_month = month - 1
            prev_predictions = df[df['Month_Val'] == prev_month][['Customer_ID', 'pred']].rename(columns={'pred': 'target_lag1'})
            
            # Map predictions
            pred_map = dict(zip(prev_predictions['Customer_ID'], prev_predictions['target_lag1']))
            
            # Find indices of current month rows
            curr_indices = df[df['Month_Val'] == month].index
            
            # Map values back into target_lag1
            df.loc[curr_indices, 'target_lag1'] = df.loc[curr_indices, 'Customer_ID'].map(pred_map)
            df['target_lag1'] = df['target_lag1'].astype('category')
            
        # 2. Extract current month rows and run model prediction
        curr_df = df[df['Month_Val'] == month]
        X = curr_df[features]
        
        y_probs = model.predict(X)
        y_preds = np.argmax(y_probs, axis=1)
        
        # Save predictions
        df.loc[curr_df.index, 'pred'] = y_preds
        
    return df
