import pandas as pd

df = pd.read_csv("Data/train.csv", low_memory=False)
print("Number of unique customers:", df['Customer_ID'].nunique())
print("Value counts of row count per customer:")
print(df['Customer_ID'].value_counts().value_counts())

print("\nMonths list:")
print(df['Month'].value_counts())

print("\nGrouped by customer and month:")
print(df.groupby('Customer_ID')['Month'].apply(list).head(5))
