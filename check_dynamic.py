import pandas as pd
import numpy as np

df = pd.read_csv("Data/train.csv", low_memory=False)

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

df['Num_of_Loan_clean'] = df['Num_of_Loan'].apply(clean_int)
df['Num_of_Delayed_Payment_clean'] = df['Num_of_Delayed_Payment'].apply(clean_int)
df['Changed_Credit_Limit_clean'] = df['Changed_Credit_Limit'].apply(clean_float)
df['Outstanding_Debt_clean'] = df['Outstanding_Debt'].apply(clean_float)
df['Amount_invested_monthly_clean'] = df['Amount_invested_monthly'].apply(clean_float)
df['Monthly_Balance_clean'] = df['Monthly_Balance'].apply(clean_float)

print("Check uniqueness of num_loans:")
print(df.groupby('Customer_ID')['Num_of_Loan_clean'].nunique().value_counts())

print("\nValue counts for Num_of_Loan_clean:")
print(df['Num_of_Loan_clean'].value_counts().sort_index().head(20))
print(df['Num_of_Loan_clean'].value_counts().sort_index().tail(10))

print("\nValue counts for Num_of_Delayed_Payment_clean:")
print(df['Num_of_Delayed_Payment_clean'].value_counts().sort_index().head(10))
print(df['Num_of_Delayed_Payment_clean'].value_counts().sort_index().tail(10))

print("\nStatistics for clean variables:")
print(df[['Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan_clean', 
          'Delay_from_due_date', 'Num_of_Delayed_Payment_clean', 'Changed_Credit_Limit_clean', 
          'Outstanding_Debt_clean', 'Amount_invested_monthly_clean', 'Monthly_Balance_clean']].describe())
