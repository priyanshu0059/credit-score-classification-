import pandas as pd
import numpy as np
import os
import joblib
from src.preprocessing import preprocess_data
from src.features import apply_customer_profiles, compute_lags_and_diffs
from src.model import sequential_predict

def run_predictions():
    print("=== Step 1: Loading Datasets & Artifacts ===")
    train_df_raw = pd.read_csv("Data/train.csv", low_memory=False)
    test_df_raw = pd.read_csv("Data/test.csv", low_memory=False)
    
    profiles = pd.read_csv("model_artifacts/customer_profiles.csv")
    model = joblib.load("model_artifacts/lightgbm_model.joblib")
    features_list, cat_features = joblib.load("model_artifacts/features_info.joblib")
    
    print(f"Train raw shape: {train_df_raw.shape}")
    print(f"Test raw shape: {test_df_raw.shape}")
    
    print("\n=== Step 2: Running Combined Preprocessing ===")
    train_clean, test_clean = preprocess_data(train_df_raw, test_df_raw)
    
    # We concatenate train and test to compute sequential inputs lags properly
    print("\n=== Step 3: Computing Input Lags & Diffs ===")
    # Target values from train are needed to initialize September's target_lag1
    target_map = {'Poor': 0, 'Standard': 1, 'Good': 2}
    train_clean['target'] = train_clean['Credit_Score'].map(target_map)
    test_clean['target'] = np.nan
    
    combined = pd.concat([train_clean, test_clean], axis=0).reset_index(drop=True)
    combined = compute_lags_and_diffs(combined)
    
    # Initialize target_lag1 in the combined dataframe
    combined['target_lag1'] = combined.groupby('Customer_ID')['target'].shift(1)
    
    # Merge customer profiles
    print("\n=== Step 4: Merging Saved Customer Profiles ===")
    combined_features = apply_customer_profiles(combined, profiles)
    
    # Fillna only for numerical columns (handling category columns appropriately)
    num_cols_to_fill = combined_features.select_dtypes(exclude=['category']).columns
    combined_features[num_cols_to_fill] = combined_features[num_cols_to_fill].fillna(0)
    
    # Now extract the test subset (months 9 to 12)
    test_subset = combined_features[combined_features['Month_Val'] >= 9].copy()
    
    # September (Month 9) target_lag1 should be August's true score
    # Let's verify September (Month 9) has target_lag1 set correctly.
    # Yes, since combined['target_lag1'] shifts the 'target' column, and for Month 9 the shift(1) points to Month 8 (August) target in train!
    # Let's cast target_lag1 to category
    test_subset['target_lag1'] = test_subset['target_lag1'].astype('category')
    
    print("\n=== Step 5: Performing Sequential Model Inference ===")
    # Auto-regressively predict September to December
    predictions_df = sequential_predict(
        model, 
        test_subset, 
        start_month=9, 
        end_month=12, 
        features=features_list, 
        categorical_features=cat_features
    )
    
    # Recast numeric predict labels back to string representations
    inv_target_map = {0: 'Poor', 1: 'Standard', 2: 'Good'}
    predictions_df['Credit_Score'] = predictions_df['pred'].map(inv_target_map)
    
    print("\n=== Step 6: Creating Submissions ===")
    # First save the full predictions with features
    full_output_path = "Data/test_predictions_full.csv"
    predictions_df.to_csv(full_output_path, index=False)
    print(f"Saved full predictions with features code to {full_output_path}")
    
    # Save the simple submission format (ID and Credit_Score)
    submission = predictions_df[['ID', 'Credit_Score']].copy()
    sub_path = "Data/submission.csv"
    submission.to_csv(sub_path, index=False)
    print(f"Saved Kaggle-format submission to {sub_path}")
    
    print("\nPrediction value counts statistics:")
    print(submission['Credit_Score'].value_counts())
    
    print("\n=== Inference Pipeline Completed ===")

if __name__ == "__main__":
    run_predictions()
