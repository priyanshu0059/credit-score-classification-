import pandas as pd
import numpy as np
import re

MONTH_MAP = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

LOAN_TYPES = [
    'Payday Loan', 'Credit-Builder Loan', 'Not Specified', 'Home Equity Loan',
    'Student Loan', 'Mortgage Loan', 'Personal Loan', 'Debt Consolidation Loan', 'Auto Loan'
]

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

def clean_int(val):
    if pd.isna(val) or val is np.nan: return np.nan
    val_str = str(val).strip().replace('_', '')
    if val_str == '' or val_str == 'nan': return np.nan
    try: return int(val_str)
    except: return np.nan

def clean_float(val):
    if pd.isna(val) or val is np.nan: return np.nan
    val_str = str(val).strip().replace('_', '')
    if '__' in val_str:
        val_str = val_str.replace('__', '')
    if val_str == '' or val_str == 'nan': return np.nan
    try: return float(val_str)
    except: return np.nan

def impute_categorical_by_customer_mode(df, col_name, val_to_nan, default_val):
    df[col_name] = df[col_name].astype(str).str.strip().replace(val_to_nan, np.nan)
    # Group by Customer_ID and get mode
    mode_per_cust = df.groupby('Customer_ID')[col_name].apply(
        lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else default_val
    )
    df[col_name] = df[col_name].fillna(df['Customer_ID'].map(mode_per_cust)).fillna(default_val)
    return df

def preprocess_data(df_train, df_test=None):
    """
    Concatenates, cleans, and imputes both datasets together to ensure perfect customer alignment.
    """
    # Create indicators
    df_train = df_train.copy()
    df_train['is_test'] = 0
    
    if df_test is not None:
        df_test = df_test.copy()
        df_test['is_test'] = 1
        df_test['Credit_Score'] = np.nan
        df = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
    else:
        df = df_train
        
    df['Month_Val'] = df['Month'].map(MONTH_MAP)
    df['History_Age_Months'] = df['Credit_History_Age'].apply(parse_history_age)
    
    # Cleaning Numeric Columns
    df['Age_clean'] = df['Age'].apply(clean_int)
    df['Annual_Income_clean'] = df['Annual_Income'].apply(clean_float)
    df['Monthly_Inhand_Salary_clean'] = df['Monthly_Inhand_Salary'].apply(clean_float)
    df['Num_Bank_Accounts_clean'] = df['Num_Bank_Accounts'].apply(clean_int)
    df['Num_Credit_Card_clean'] = df['Num_Credit_Card'].apply(clean_int)
    df['Interest_Rate_clean'] = df['Interest_Rate'].apply(clean_int)
    df['Num_of_Loan_clean'] = df['Num_of_Loan'].apply(clean_int)
    df['Num_of_Delayed_Payment_clean'] = df['Num_of_Delayed_Payment'].apply(clean_float)
    df['Changed_Credit_Limit_clean'] = df['Changed_Credit_Limit'].apply(clean_float)
    df['Outstanding_Debt_clean'] = df['Outstanding_Debt'].apply(clean_float)
    df['Amount_invested_monthly_clean'] = df['Amount_invested_monthly'].apply(clean_float)
    df['Monthly_Balance_clean'] = df['Monthly_Balance'].apply(clean_float)
    df['Num_Credit_Inquiries_clean'] = df['Num_Credit_Inquiries'].apply(clean_float)
    
    # Cap / filter outliers
    df.loc[(df['Age_clean'] < 18) | (df['Age_clean'] > 80), 'Age_clean'] = np.nan
    df.loc[(df['Num_Bank_Accounts_clean'] < 0) | (df['Num_Bank_Accounts_clean'] > 15), 'Num_Bank_Accounts_clean'] = np.nan
    df.loc[(df['Num_Credit_Card_clean'] < 0) | (df['Num_Credit_Card_clean'] > 15), 'Num_Credit_Card_clean'] = np.nan
    df.loc[(df['Interest_Rate_clean'] < 1) | (df['Interest_Rate_clean'] > 34), 'Interest_Rate_clean'] = np.nan
    df.loc[(df['Num_of_Loan_clean'] < 0) | (df['Num_of_Loan_clean'] > 15), 'Num_of_Loan_clean'] = np.nan
    df['Delay_from_due_date_clean'] = df['Delay_from_due_date'].clip(lower=0)
    df.loc[(df['Num_of_Delayed_Payment_clean'] < 0) | (df['Num_of_Delayed_Payment_clean'] > 28), 'Num_of_Delayed_Payment_clean'] = np.nan
    df.loc[(df['Num_Credit_Inquiries_clean'] < 0) | (df['Num_Credit_Inquiries_clean'] > 20), 'Num_Credit_Inquiries_clean'] = np.nan
    df.loc[df['Monthly_Balance_clean'] < 0, 'Monthly_Balance_clean'] = np.nan
    
    # Impute numeric metrics with customer specific medians
    cols_to_median = [
        'Age_clean', 'Annual_Income_clean', 'Monthly_Inhand_Salary_clean',
        'Num_Bank_Accounts_clean', 'Num_Credit_Card_clean', 'Interest_Rate_clean',
        'Num_of_Loan_clean', 'Num_of_Delayed_Payment_clean', 'Changed_Credit_Limit_clean',
        'Num_Credit_Inquiries_clean', 'Outstanding_Debt_clean', 'Amount_invested_monthly_clean',
        'Monthly_Balance_clean'
    ]
    customer_medians = df.groupby('Customer_ID')[cols_to_median].transform('median')
    for col in cols_to_median:
        df[col] = df[col].fillna(customer_medians[col]).fillna(df[col].median())
        
    # Reconstruct History Age (linear regression)
    df['Base_History_Age'] = df['History_Age_Months'] - (df['Month_Val'] - 1)
    customer_base_history = df.groupby('Customer_ID')['Base_History_Age'].transform('median')
    global_median_base = df['Base_History_Age'].median()
    df['History_Age_Months_Imputed'] = customer_base_history.fillna(global_median_base) + (df['Month_Val'] - 1)
    
    # Categoricals cleaning
    df = impute_categorical_by_customer_mode(df, 'Occupation', '_______', 'Unknown')
    df = impute_categorical_by_customer_mode(df, 'Credit_Mix', '_', 'Unknown')
    df = impute_categorical_by_customer_mode(df, 'Payment_of_Min_Amount', 'NM', 'Unknown')
    df = impute_categorical_by_customer_mode(df, 'Payment_Behaviour', '!@9#%8', 'Unknown')
    df = impute_categorical_by_customer_mode(df, 'Type_of_Loan', 'nan', 'No Loan')
    
    # Loan indicators
    for loan in LOAN_TYPES:
        col_name = 'Loan_' + loan.replace(' ', '_').replace('-', '_')
        df[col_name] = df['Type_of_Loan'].apply(lambda x: 1 if loan in str(x) else 0)
        
    # Label encode / Category cast
    cat_cols = ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour']
    for col in cat_cols:
        df[col] = df[col].astype('category')
        
    # Standardize sort
    df = df.sort_values(['Customer_ID', 'Month_Val']).reset_index(drop=True)
    
    if df_test is not None:
        train_processed = df[df['is_test'] == 0].drop(columns=['is_test'])
        test_processed = df[df['is_test'] == 1].drop(columns=['is_test'])
        return train_processed, test_processed
    else:
        return df.drop(columns=['is_test'])
