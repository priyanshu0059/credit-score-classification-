import pandas as pd
import numpy as np

df = pd.read_csv("Data/train.csv", low_memory=False)

# Let's clean the columns slightly to see if they are invariant
# First define cleaner of object columns to numbers
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

df['Age_clean'] = df['Age'].apply(clean_int)
df['Annual_Income_clean'] = df['Annual_Income'].apply(clean_float)

print("Check uniqueness per Customer_ID for some fields:")
for col in ['Name', 'SSN', 'Age_clean', 'Occupation', 'Annual_Income_clean', 'Monthly_Inhand_Salary']:
    nunique = df.groupby('Customer_ID')[col].nunique(dropna=True)
    print(f"Column {col} nunique per customer value counts:")
    print(nunique.value_counts())
