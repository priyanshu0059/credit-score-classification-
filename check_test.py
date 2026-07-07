import pandas as pd

df_train = pd.read_csv("Data/train.csv", low_memory=False)
df_test = pd.read_csv("Data/test.csv", low_memory=False)

print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)

print("\nTest columns:")
print(df_test.columns)

print("\nNumber of unique customers in Test:", df_test['Customer_ID'].nunique())
print("Overlap in Customer_ID between Train and Test:")
overlap = len(set(df_train['Customer_ID']).intersection(set(df_test['Customer_ID'])))
print("Overlap count:", overlap)

print("\nValue counts of row count per customer in Test:")
print(df_test['Customer_ID'].value_counts().value_counts())

print("\nMonths list in Test:")
print(df_test['Month'].value_counts())

print("\nDoes test.csv contain Credit_Score?")
print("Credit_Score in test.csv:", 'Credit_Score' in df_test.columns)

print("\nFirst 3 rows of test.csv:")
print(df_test.head(3).to_string())
