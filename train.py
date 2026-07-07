import pandas as pd
import numpy as np
import os
import joblib
from src.preprocessing import preprocess_data
from src.features import compute_customer_profiles, apply_customer_profiles, compute_lags_and_diffs
from src.model import get_features_list, train_model

def run_training():
    print("=== Step 1: Loading Training Data ===")
    train_df_raw = pd.read_csv("Data/train.csv", low_memory=False)
    print(f"Raw data shape: {train_df_raw.shape}")
    
    print("\n=== Step 2: Running Preprocessing and Imputation ===")
    df_clean = preprocess_data(train_df_raw)
    print(f"Cleaned data shape: {df_clean.shape}")
    
    print("\n=== Step 3: Computing Input Lags & Diffs ===")
    df_lags = compute_lags_and_diffs(df_clean)
    
    # Target label mapping
    target_map = {'Poor': 0, 'Standard': 1, 'Good': 2}
    df_lags['target'] = df_lags['Credit_Score'].map(target_map)
    
    # Calculate target_lag1 (credit score of the previous month)
    df_lags['target_lag1'] = df_lags.groupby('Customer_ID')['target'].shift(1)
    # Cast to category for LightGBM native categoricals
    df_lags['target_lag1'] = df_lags['target_lag1'].astype('category')
    
    print("\n=== Step 4: Computing & Saving Customer Profiles ===")
    # Calculate profiles over the ENTIRE training timeline (months 1-8)
    profiles = compute_customer_profiles(df_lags)
    os.makedirs('model_artifacts', exist_ok=True)
    profiles.to_csv("model_artifacts/customer_profiles.csv", index=False)
    print(f"Saved profiles for {profiles['Customer_ID'].nunique()} customers to model_artifacts/customer_profiles.csv")
    
    # Merge profiles back
    df_features = apply_customer_profiles(df_lags, profiles)
    
    # Fillna only for numerical columns (handling category columns appropriately)
    num_cols_to_fill = df_features.select_dtypes(exclude=['category']).columns
    df_features[num_cols_to_fill] = df_features[num_cols_to_fill].fillna(0)
    
    print("\n=== Step 5: Preparing Training Set ===")
    # For training, we select months 2 to 8 (where target_lag1 is populated)
    train_subset = df_features[(df_features['Month_Val'] >= 2) & (df_features['Month_Val'] <= 8)].copy()
    
    # Construct features list using the profiles dataframe columns
    profile_columns = [col for col in profiles.columns if col != 'Customer_ID']
    features_list = get_features_list(profile_columns, include_target_lag=True)
    
    cat_features = ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour', 'target_lag1']
    
    X_train = train_subset[features_list]
    y_train = train_subset['target']
    
    print(f"X_train shape: {X_train.shape}")
    print(f"Number of training features: {len(features_list)}")
    print(f"Categorical features: {cat_features}")
    
    print("\n=== Step 6: Training LightGBM Model ===")
    # We train for 230 boosting rounds as determined by our validation experiments
    model = train_model(X_train, y_train, cat_features, num_boost_round=230)
    
    # Save model
    model_path = "model_artifacts/lightgbm_model.joblib"
    joblib.dump(model, model_path)
    print(f"Optimized LightGBM model saved to {model_path}")
    
    # Save the list of features for inference matching
    joblib.dump((features_list, cat_features), "model_artifacts/features_info.joblib")
    print("Features info saved successfully!")
    print("\n=== Training Completed Successfully ===")

if __name__ == "__main__":
    run_training()
