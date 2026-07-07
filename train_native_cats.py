import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import re

print("Loading data...")
df = pd.read_csv("Data/train.csv", low_memory=False)

month_map = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8
}
df['Month_Val'] = df['Month'].map(month_map)

# Helper function to parse Credit_History_Age
def parse_history_age(val):
    if pd.isna(val) or not isinstance(val, str):
        return np.nan
    val = val.strip()
    match = re.match(r'(\d+)\s+Years\s+and\s+(\d+)\s+Months', val)
    if match:
        years = int(match.group(1))
        months = int(match.group(2))
        return years * 12 + months
    return np.nan

df['History_Age_Months'] = df['Credit_History_Age'].apply(parse_history_age)

# Define clean helpers
def clean_int(val):
    if pd.isna(val): return np.nan
    val_str = str(val).strip().replace('_', '')
    if val_str == '' or val_str == 'nan': return np.nan
    try: return int(val_str)
    except: return np.nan

def clean_float(val):
    if pd.isna(val): return np.nan
    val_str = str(val).strip().replace('_', '')
    if '__' in val_str:
        val_str = val_str.replace('__', '')
    if val_str == '' or val_str == 'nan': return np.nan
    try: return float(val_str)
    except: return np.nan

# Basic cleaning
df['Age_clean'] = df['Age'].apply(clean_int)
df['Annual_Income_clean'] = df['Annual_Income'].apply(clean_float)
df['Monthly_Inhand_Salary_clean'] = df['Monthly_Inhand_Salary'].apply(clean_float)
df['Num_Bank_Accounts_clean'] = df['Num_Bank_Accounts'].apply(clean_int)
df['Num_Credit_Card_clean'] = df['Num_Credit_Card_clean'] = df['Num_Credit_Card'].apply(clean_int)
df['Interest_Rate_clean'] = df['Interest_Rate'].apply(clean_int)
df['Num_of_Loan_clean'] = df['Num_of_Loan'].apply(clean_int)
df['Num_of_Delayed_Payment_clean'] = df['Num_of_Delayed_Payment'].apply(clean_float)
df['Changed_Credit_Limit_clean'] = df['Changed_Credit_Limit'].apply(clean_float)
df['Outstanding_Debt_clean'] = df['Outstanding_Debt'].apply(clean_float)
df['Amount_invested_monthly_clean'] = df['Amount_invested_monthly'].apply(clean_float)
df['Monthly_Balance_clean'] = df['Monthly_Balance'].apply(clean_float)
df['Num_Credit_Inquiries_clean'] = df['Num_Credit_Inquiries'].apply(clean_float)

# Capping outliers:
df.loc[(df['Age_clean'] < 18) | (df['Age_clean'] > 80), 'Age_clean'] = np.nan
df.loc[(df['Num_Bank_Accounts_clean'] < 0) | (df['Num_Bank_Accounts_clean'] > 15), 'Num_Bank_Accounts_clean'] = np.nan
df.loc[(df['Num_Credit_Card_clean'] < 0) | (df['Num_Credit_Card_clean'] > 15), 'Num_Credit_Card_clean'] = np.nan
df.loc[(df['Interest_Rate_clean'] < 1) | (df['Interest_Rate_clean'] > 34), 'Interest_Rate_clean'] = np.nan
df.loc[(df['Num_of_Loan_clean'] < 0) | (df['Num_of_Loan_clean'] > 15), 'Num_of_Loan_clean'] = np.nan
df['Delay_from_due_date_clean'] = df['Delay_from_due_date'].clip(lower=0)
df.loc[(df['Num_of_Delayed_Payment_clean'] < 0) | (df['Num_of_Delayed_Payment_clean'] > 28), 'Num_of_Delayed_Payment_clean'] = np.nan
df.loc[(df['Num_Credit_Inquiries_clean'] < 0) | (df['Num_Credit_Inquiries_clean'] > 20), 'Num_Credit_Inquiries_clean'] = np.nan
df.loc[df['Monthly_Balance_clean'] < 0, 'Monthly_Balance_clean'] = np.nan

# Imputation using Customer_ID medians
customer_medians = df.groupby('Customer_ID')[[
    'Age_clean', 'Annual_Income_clean', 'Monthly_Inhand_Salary_clean',
    'Num_Bank_Accounts_clean', 'Num_Credit_Card_clean', 'Interest_Rate_clean',
    'Num_of_Loan_clean', 'Num_of_Delayed_Payment_clean', 'Changed_Credit_Limit_clean',
    'Num_Credit_Inquiries_clean', 'Outstanding_Debt_clean', 'Amount_invested_monthly_clean',
    'Monthly_Balance_clean'
]].transform('median')

for col in customer_medians.columns:
    df[col] = df[col].fillna(customer_medians[col]).fillna(df[col].median())

# Linear history age regression
df['Base_History_Age'] = df['History_Age_Months'] - (df['Month_Val'] - 1)
customer_base_history = df.groupby('Customer_ID')['Base_History_Age'].transform('median')
global_median_base = df['Base_History_Age'].median()
df['History_Age_Months_Imputed'] = customer_base_history.fillna(global_median_base) + (df['Month_Val'] - 1)

# Categorical clean and impute
def impute_categorical_by_customer_mode(df, col_name, val_to_nan, default_val):
    df[col_name] = df[col_name].astype(str).str.strip().replace(val_to_nan, np.nan)
    mode_per_cust = df.groupby('Customer_ID')[col_name].apply(
        lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else default_val
    )
    df[col_name] = df[col_name].fillna(df['Customer_ID'].map(mode_per_cust)).fillna(default_val)
    return df

df = impute_categorical_by_customer_mode(df, 'Occupation', '_______', 'Unknown')
df = impute_categorical_by_customer_mode(df, 'Credit_Mix', '_', 'Unknown')
df = impute_categorical_by_customer_mode(df, 'Payment_of_Min_Amount', 'NM', 'Unknown')
df = impute_categorical_by_customer_mode(df, 'Payment_Behaviour', '!@9#%8', 'Unknown')

# Feature Extraction: Type of Loan binary flags
loan_types = [
    'Payday Loan', 'Credit-Builder Loan', 'Not Specified', 'Home Equity Loan',
    'Student Loan', 'Mortgage Loan', 'Personal Loan', 'Debt Consolidation Loan', 'Auto Loan'
]
df = impute_categorical_by_customer_mode(df, 'Type_of_Loan', 'nan', 'No Loan')
for loan in loan_types:
    col_name = 'Loan_' + loan.replace(' ', '_').replace('-', '_')
    df[col_name] = df['Type_of_Loan'].apply(lambda x: 1 if loan in str(x) else 0)

# Convert categorical columns to category dtypes
cat_cols = ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour']
for col in cat_cols:
    df[col] = df[col].astype('category')

# Map target to int
target_map = {'Poor': 0, 'Standard': 1, 'Good': 2}
df['target'] = df['Credit_Score'].map(target_map)

# SORT BEFORE LAGS
df = df.sort_values(['Customer_ID', 'Month_Val']).reset_index(drop=True)

# Compute lags and diffs of inputs
lag_cols = [
    'Outstanding_Debt_clean', 'Delay_from_due_date_clean', 'Num_of_Delayed_Payment_clean',
    'Changed_Credit_Limit_clean', 'Num_Credit_Inquiries_clean', 'Monthly_Balance_clean',
    'Amount_invested_monthly_clean', 'Total_EMI_per_month'
]

for col in lag_cols:
    df[f"{col}_lag1"] = df.groupby('Customer_ID')[col].shift(1)
    df[f"{col}_diff1"] = df[col] - df[f"{col}_lag1"]
    df[f"{col}_lag2"] = df.groupby('Customer_ID')[col].shift(2)
    df[f"{col}_diff2"] = df[col] - df[f"{col}_lag2"]

# target lag feature
df['target_lag1'] = df.groupby('Customer_ID')['target'].shift(1)
df['target_lag1'] = df['target_lag1'].astype('category')

# Split train & validation
train_df = df[(df['Month_Val'] >= 2) & (df['Month_Val'] <= 6)].copy()
val_df = df[df['Month_Val'] > 6].copy()

# Profile features computed ONLY on training partition
cols_to_profile = [
    'Outstanding_Debt_clean', 'Delay_from_due_date_clean', 'Num_of_Delayed_Payment_clean',
    'Changed_Credit_Limit_clean', 'Num_Credit_Inquiries_clean', 'Monthly_Balance_clean'
]
train_df_profiling = df[df['Month_Val'] <= 6]
train_profiles = train_df_profiling.groupby('Customer_ID')[cols_to_profile].agg(['mean', 'std', 'min', 'max'])
train_profiles.columns = [f"{col}_{stat}" for col, stat in train_profiles.columns]
train_profiles = train_profiles.reset_index()

train_df = train_df.merge(train_profiles, on='Customer_ID', how='left')
val_df = val_df.merge(train_profiles, on='Customer_ID', how='left')

# Fillna only for numerical columns
num_cols_to_fill = train_df.select_dtypes(exclude=['category']).columns
train_df[num_cols_to_fill] = train_df[num_cols_to_fill].fillna(0)

val_cols_to_fill = val_df.select_dtypes(exclude=['category']).columns
val_df[val_cols_to_fill] = val_df[val_cols_to_fill].fillna(0)

# Feature lists
numeric_features = [
    'Age_clean', 'Annual_Income_clean', 'Monthly_Inhand_Salary_clean',
    'Num_Bank_Accounts_clean', 'Num_Credit_Card_clean', 'Interest_Rate_clean',
    'Num_of_Loan_clean', 'Delay_from_due_date_clean', 'Num_of_Delayed_Payment_clean',
    'Changed_Credit_Limit_clean', 'Num_Credit_Inquiries_clean', 'Outstanding_Debt_clean',
    'Total_EMI_per_month', 'Amount_invested_monthly_clean', 'Monthly_Balance_clean',
    'History_Age_Months_Imputed'
]

profile_cols = [f"{col}_{stat}" for col in cols_to_profile for stat in ['mean', 'std', 'min', 'max']]

lag_features = []
for col in lag_cols:
    lag_features.extend([f"{col}_lag1", f"{col}_diff1", f"{col}_lag2", f"{col}_diff2"])

loan_features = ['Loan_' + loan.replace(' ', '_').replace('-', '_') for loan in loan_types]

categorical_features = cat_cols + ['target_lag1']

features = numeric_features + profile_cols + lag_features + loan_features + categorical_features

X_train = train_df[features]
y_train = train_df['target']

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)

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

print("Training model with native categoricals...")
model = lgb.train(
    params,
    train_data,
    num_boost_round=230
)

# Sequential prediction
print("Performing sequential validation...")
val_df = val_df.sort_values(['Customer_ID', 'Month_Val']).reset_index(drop=True)
val_7 = val_df[val_df['Month_Val'] == 7].copy()
val_8 = val_df[val_df['Month_Val'] == 8].copy()

# Month 7
X_val_7 = val_7[features]
y_pred_probs_7 = model.predict(X_val_7)
y_pred_7 = np.argmax(y_pred_probs_7, axis=1)
val_7['pred'] = y_pred_7

# Map Month 7 prediction to Month 8 lag
pred_dict = dict(zip(val_7['Customer_ID'], val_7['pred']))
val_8['target_lag1'] = val_8['Customer_ID'].map(pred_dict).fillna(1.0) # Standard count fallback
val_8['target_lag1'] = val_8['target_lag1'].astype('category')

# Month 8
X_val_8 = val_8[features]
y_pred_probs_8 = model.predict(X_val_8)
y_pred_8 = np.argmax(y_pred_probs_8, axis=1)
val_8['pred'] = y_pred_8

# Combine
val_combined = pd.concat([val_7, val_8]).sort_values(['Customer_ID', 'Month_Val'])
y_val_combined = val_combined['target']
y_pred_combined = val_combined['pred']

acc = accuracy_score(y_val_combined, y_pred_combined)
macro_f1 = f1_score(y_val_combined, y_pred_combined, average='macro')
print(f"\nNative Cats Validation Accuracy: {acc:.5f}")
print(f"Native Cats Validation Macro F1: {macro_f1:.5f}")
