import pandas as pd
from collections import Counter

df = pd.read_csv("Data/train.csv", low_memory=False)

loan_counts = Counter()
for val in df['Type_of_Loan'].dropna():
    # Split by comma
    parts = val.split(',')
    for part in parts:
        part = part.strip()
        # Remove leading "and "
        if part.startswith("and "):
            part = part[4:].strip()
        # Simple cleanup
        if part:
            loan_counts[part] += 1

print("Loan type frequencies:")
for loan_type, count in loan_counts.most_common():
    print(f"  {loan_type}: {count}")
