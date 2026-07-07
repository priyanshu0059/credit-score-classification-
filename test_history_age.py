import pandas as pd
import numpy as np
import re

df = pd.read_csv("Data/train.csv", low_memory=False)

month_map = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}
df['Month_Val'] = df['Month'].map(month_map)

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

# Let's inspect a customer
cust_example = df[df['Customer_ID'] == 'CUS_0xd40'][['Month', 'Month_Val', 'Credit_History_Age', 'History_Age_Months']]
print("CUS_0xd40 Credit History Age progression:")
print(cust_example)

# Calculate Base Months per customer
df['Base_History_Age'] = df['History_Age_Months'] - (df['Month_Val'] - 1)
base_per_cust = df.groupby('Customer_ID')['Base_History_Age'].median()

# Fill in and print
df['Imputed_History_Age'] = df['Customer_ID'].map(base_per_cust) + (df['Month_Val'] - 1)
cust_example_imputed = df[df['Customer_ID'] == 'CUS_0xd40'][['Month', 'Month_Val', 'Credit_History_Age', 'History_Age_Months', 'Imputed_History_Age']]
print("\nImputed progression for CUS_0xd40:")
print(cust_example_imputed)

# Let's check how many customers have no valid credit history age at all
missing_any = base_per_cust.isna().sum()
print(f"\nNumber of customers with NO valid credit history age: {missing_any} out of {df['Customer_ID'].nunique()}")
