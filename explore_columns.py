import pandas as pd
import numpy as np
import re

print("Loading train.csv for detail exploration...")
df = pd.read_csv("Data/train.csv", low_memory=False)

def check_col_non_numeric(col):
    non_num = df[col].astype(str).apply(lambda x: re.search(r'[^0-9\.\-]', x) is not None)
    print(f"\nColumn [{col}] - sample of unique non-numeric values (excluding simple floats):")
    non_num_vals = df.loc[non_num, col].unique()
    print(non_num_vals[:15])
    print(f"Total rows with non-numeric chars: {non_num.sum()} out of {len(df)}")

cols_to_check = [
    'Age', 'Annual_Income', 'Num_of_Clean_Loan', 'Num_of_Loan', 'Num_of_Delayed_Payment',
    'Changed_Credit_Limit', 'Outstanding_Debt', 'Amount_invested_monthly', 'Monthly_Balance'
]

# Adjust cols to check based on what exists
cols_to_check = [c for c in cols_to_check if c in df.columns]

for col in cols_to_check:
    check_col_non_numeric(col)

print("\nOccupation unique values:")
print(df['Occupation'].unique()[:20])

print("\nCredit_Mix unique values:")
print(df['Credit_Mix'].unique())

print("\nPayment_of_Min_Amount unique values:")
print(df['Payment_of_Min_Amount'].unique())

print("\nPayment_Behaviour unique values:")
print(df['Payment_Behaviour'].unique())

print("\nCredit_History_Age unique values:")
print(df['Credit_History_Age'].unique()[:10])

print("\nType_of_Loan unique values:")
print(df['Type_of_Loan'].dropna().unique()[:5])
