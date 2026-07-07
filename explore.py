import pandas as pd
import numpy as np

# Load a sample or full dataset
print("Loading train.csv...")
df = pd.read_csv("Data/train.csv", low_memory=False)
print("Data shape:", df.shape)

print("\nColumns and Info:")
print(df.info())

print("\nNull values count:")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\nValue counts for Credit_Score:")
if 'Credit_Score' in df.columns:
    print(df['Credit_Score'].value_counts(dropna=False))
else:
    print("Credit_Score column not found!")

print("\nFirst 3 rows:")
print(df.head(3).to_string())
