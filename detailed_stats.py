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

# Create cleaned columns
df['Age_clean'] = df['Age'].apply(clean_int)
df['Annual_Income_clean'] = df['Annual_Income'].apply(clean_float)
df['Num_of_Loan_clean'] = df['Num_of_Loan'].apply(clean_int)
df['Num_of_Delayed_Payment_clean'] = df['Num_of_Delayed_Payment'].apply(clean_int)
df['Changed_Credit_Limit_clean'] = df['Changed_Credit_Limit'].apply(clean_float)
df['Outstanding_Debt_clean'] = df['Outstanding_Debt'].apply(clean_float)
df['Amount_invested_monthly_clean'] = df['Amount_invested_monthly'].apply(clean_float)
df['Monthly_Balance_clean'] = df['Monthly_Balance'].apply(clean_float)
df['Num_Credit_Inquiries_clean'] = df['Num_Credit_Inquiries'].apply(clean_float)

numerical_cols = [
    'Age_clean', 'Annual_Income_clean', 'Monthly_Inhand_Salary', 
    'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan_clean', 
    'Delay_from_due_date', 'Num_of_Delayed_Payment_clean', 'Changed_Credit_Limit_clean', 
    'Num_Credit_Inquiries_clean', 'Outstanding_Debt_clean', 'Total_EMI_per_month', 
    'Amount_invested_monthly_clean', 'Monthly_Balance_clean'
]

for col in numerical_cols:
    s = df[col].dropna()
    print(f"\n{col} Statistics:")
    print(f"  Count of non-null: {len(s)}")
    print(f"  Min: {s.min()}, Max: {s.max()}")
    print(f"  Percentiles (1%, 5%, 25%, 50%, 75%, 95%, 99%):")
    print(f"  {np.percentile(s, [1, 5, 25, 50, 75, 95, 99])}")
