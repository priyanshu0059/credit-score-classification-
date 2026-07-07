import pandas as pd
import numpy as np

PROFILE_COLS = [
    'Outstanding_Debt_clean', 'Delay_from_due_date_clean', 'Num_of_Delayed_Payment_clean',
    'Changed_Credit_Limit_clean', 'Num_Credit_Inquiries_clean', 'Monthly_Balance_clean'
]

LAG_COLS = [
    'Outstanding_Debt_clean', 'Delay_from_due_date_clean', 'Num_of_Delayed_Payment_clean',
    'Changed_Credit_Limit_clean', 'Num_Credit_Inquiries_clean', 'Monthly_Balance_clean',
    'Amount_invested_monthly_clean', 'Total_EMI_per_month'
]

def compute_customer_profiles(train_df):
    """
    Computes profile stats (mean, std, min, max) for each customer using the specified historical context.
    """
    profiles = train_df.groupby('Customer_ID')[PROFILE_COLS].agg(['mean', 'std', 'min', 'max'])
    profiles.columns = [f"{col}_{stat}" for col, stat in profiles.columns]
    profiles = profiles.reset_index()
    return profiles

def apply_customer_profiles(target_df, profiles):
    """
    Merges customer profiles onto the target df.
    """
    merged = target_df.merge(profiles, on='Customer_ID', how='left')
    
    # Fill standard deviation with 0 if missing (e.g. single records or no profile)
    std_cols = [c for c in merged.columns if '_std' in c]
    merged[std_cols] = merged[std_cols].fillna(0)
    
    # Fill remaining profile features with overall median if missing
    profile_features = [col for col in Target_Features_List(profiles.columns) if col != 'Customer_ID']
    for col in profile_features:
        if col in merged.columns:
            merged[col] = merged[col].fillna(merged[col].median())
            
    return merged

def Target_Features_List(columns):
    return list(columns)

def compute_lags_and_diffs(df):
    """
    Computes historical values (lag 1, lag 2) and their differences for dynamic metrics.
    Assumes df is sorted by Customer_ID and Month_Val.
    """
    df = df.copy()
    for col in LAG_COLS:
        # Lag 1
        df[f"{col}_lag1"] = df.groupby('Customer_ID')[col].shift(1)
        df[f"{col}_diff1"] = df[col] - df[f"{col}_lag1"]
        # Lag 2
        df[f"{col}_lag2"] = df.groupby('Customer_ID')[col].shift(2)
        df[f"{col}_diff2"] = df[col] - df[f"{col}_lag2"]
        
    return df
