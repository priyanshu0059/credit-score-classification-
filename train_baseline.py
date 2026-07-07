import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import re

print("Loading data...")
df = pd.read_csv("Data/train.csv", low_memory=False)

# Let's map months
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
df['Num_Credit_Card_clean'] = df['Num_Credit_Card'].apply(clean_int)
df['Interest_Rate_clean'] = df['Interest_Rate'].apply(clean_int)
df['Num_of_Loan_clean'] = df['Num_of_Loan'].apply(clean_int)
df['Num_of_Delayed_Payment_clean'] = df['Num_of_Delayed_Payment'].apply(clean_float) # Some delayed payment are floats
df['Changed_Credit_Limit_clean'] = df['Changed_Credit_Limit'].apply(clean_float)
df['Outstanding_Debt_clean'] = df['Outstanding_Debt'].apply(clean_float)
df['Amount_invested_monthly_clean'] = df['Amount_invested_monthly'].apply(clean_float)
df['Monthly_Balance_clean'] = df['Monthly_Balance'].apply(clean_float)
df['Num_Credit_Inquiries_clean'] = df['Num_Credit_Inquiries'].apply(clean_float)

# Capping outliers based on detailed_stats:
# Age: 18 to 80
df.loc[(df['Age_clean'] < 18) | (df['Age_clean'] > 80), 'Age_clean'] = np.nan
# Num_Bank_Accounts: 0 to 15
df.loc[(df['Num_Bank_Accounts_clean'] < 0) | (df['Num_Bank_Accounts_clean'] > 15), 'Num_Bank_Accounts_clean'] = np.nan
# Num_Credit_Card: 0 to 15
df.loc[(df['Num_Credit_Card_clean'] < 0) | (df['Num_Credit_Card_clean'] > 15), 'Num_Credit_Card_clean'] = np.nan
# Interest_Rate: 1 to 34
df.loc[(df['Interest_Rate_clean'] < 1) | (df['Interest_Rate_clean'] > 34), 'Interest_Rate_clean'] = np.nan
# Num_of_Loan: 0 to 15
df.loc[(df['Num_of_Loan_clean'] < 0) | (df['Num_of_Loan_clean'] > 15), 'Num_of_Loan_clean'] = np.nan
# Delay_from_due_date: clip at 0
df['Delay_from_due_date_clean'] = df['Delay_from_due_date'].clip(lower=0)
# Num_of_Delayed_Payment: 0 to 28
df.loc[(df['Num_of_Delayed_Payment_clean'] < 0) | (df['Num_of_Delayed_Payment_clean'] > 28), 'Num_of_Delayed_Payment_clean'] = np.nan
# Num_Credit_Inquiries: 0 to 20
df.loc[(df['Num_Credit_Inquiries_clean'] < 0) | (df['Num_Credit_Inquiries_clean'] > 20), 'Num_Credit_Inquiries_clean'] = np.nan
# Monthly_Balance: make NaN if <= -100 or huge negative
df.loc[df['Monthly_Balance_clean'] < 0, 'Monthly_Balance_clean'] = np.nan

# Imputation using Customer_ID medians
customer_medians = df.groupby('Customer_ID')[[
    'Age_clean', 'Annual_Income_clean', 'Monthly_Inhand_Salary_clean',
    'Num_Bank_Accounts_clean', 'Num_Credit_Card_clean', 'Interest_Rate_clean',
    'Num_of_Loan_clean', 'Num_of_Delayed_Payment_clean', 'Changed_Credit_Limit_clean',
    'Num_Credit_Inquiries_clean', 'Outstanding_Debt_clean', 'Amount_invested_monthly_clean',
    'Monthly_Balance_clean'
]].transform('median')

# Fill missing values with customer specific medians, then default to overall median
for col in customer_medians.columns:
    df[col] = df[col].fillna(customer_medians[col]).fillna(df[col].median())

# For credit history age, use the linear month regression
df['Base_History_Age'] = df['History_Age_Months'] - (df['Month_Val'] - 1)
customer_base_history = df.groupby('Customer_ID')['Base_History_Age'].transform('median')
global_median_base = df['Base_History_Age'].median()
df['History_Age_Months_Imputed'] = customer_base_history.fillna(global_median_base) + (df['Month_Val'] - 1)

# Categorical clean and impute
# Mode imputation helper for customer
def impute_categorical_by_customer_mode(df, col_name, val_to_nan, default_val):
    df[col_name] = df[col_name].astype(str).str.strip().replace(val_to_nan, np.nan)
    
    # Calculate mode per customer
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

# Impute Type_of_Loan by customer mode first
df = impute_categorical_by_customer_mode(df, 'Type_of_Loan', 'nan', 'No Loan')

for loan in loan_types:
    col_name = 'Loan_' + loan.replace(' ', '_').replace('-', '_')
    df[col_name] = df['Type_of_Loan'].apply(lambda x: 1 if loan in str(x) else 0)

# Label encoding for categoricals
cat_cols = ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour']
for col in cat_cols:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])

# Prepare training columns
numeric_features = [
    'Age_clean', 'Annual_Income_clean', 'Monthly_Inhand_Salary_clean',
    'Num_Bank_Accounts_clean', 'Num_Credit_Card_clean', 'Interest_Rate_clean',
    'Num_of_Loan_clean', 'Delay_from_due_date_clean', 'Num_of_Delayed_Payment_clean',
    'Changed_Credit_Limit_clean', 'Num_Credit_Inquiries_clean', 'Outstanding_Debt_clean',
    'Total_EMI_per_month', 'Amount_invested_monthly_clean', 'Monthly_Balance_clean',
    'History_Age_Months_Imputed'
]

loan_features = ['Loan_' + loan.replace(' ', '_').replace('-', '_') for loan in loan_types]
encoded_cat_features = [col + '_encoded' for col in cat_cols]

features = numeric_features + loan_features + encoded_cat_features
target = 'Credit_Score'

# Map target to int
target_map = {'Poor': 0, 'Standard': 1, 'Good': 2}
df['target'] = df[target].map(target_map)

# Split train & validation
# Train: January to June (Month_Val <= 6)
# Val: July to August (Month_Val > 6)
train_df = df[df['Month_Val'] <= 6]
val_df = df[df['Month_Val'] > 6]

X_train = train_df[features]
y_train = train_df['target']
X_val = val_df[features]
y_val = val_df['target']

print(f"Train features shape: {X_train.shape}")
print(f"Validation features shape: {X_val.shape}")

# Train LightGBM model
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 63,
    'max_depth': -1,
    'feature_fraction': 0.8,
    'verbose': -1,
    'random_state': 42
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=300,
    valid_sets=[train_data, val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
)

# Predict validation
y_pred_probs = model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)

acc = accuracy_score(y_val, y_pred)
macro_f1 = f1_score(y_val, y_pred, average='macro')
print(f"\nBaseline Validation Accuracy: {acc:.5f}")
print(f"Baseline Validation Macro F1: {macro_f1:.5f}")

print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=['Poor', 'Standard', 'Good']))
