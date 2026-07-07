import pandas as pd

df = pd.read_csv("Data/train.csv", low_memory=False)
customer_scores = df.groupby('Customer_ID')['Credit_Score'].nunique()
print("Number of unique credit scores per customer in Train:")
print(customer_scores.value_counts())

print("\nSample customers with changes in Credit_Score:")
chang_custs = customer_scores[customer_scores > 1].index
if len(chang_custs) > 0:
    sample_df = df[df['Customer_ID'].isin(chang_custs[:3])][['Customer_ID', 'Month', 'Credit_Score']]
    print(sample_df)
else:
    print("None found!")
