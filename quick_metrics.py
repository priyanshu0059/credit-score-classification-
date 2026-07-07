import pandas as pd, numpy as np, joblib, warnings; warnings.filterwarnings('ignore')
from src.preprocessing import preprocess_data
from src.features import compute_lags_and_diffs
from sklearn.metrics import accuracy_score, f1_score, classification_report

model = joblib.load('model_artifacts/lightgbm_model.joblib')
profiles = pd.read_csv('model_artifacts/customer_profiles.csv')
features_list, cat_features = joblib.load('model_artifacts/features_info.joblib')

train_raw = pd.read_csv('Data/train.csv', low_memory=False)
df_clean  = preprocess_data(train_raw)
df_lags   = compute_lags_and_diffs(df_clean)

target_map = {'Poor':0, 'Standard':1, 'Good':2}
df_lags['target'] = df_lags['Credit_Score'].map(target_map)
df_lags['target_lag1'] = df_lags.groupby('Customer_ID')['target'].shift(1).astype('category')

df_feat = df_lags.merge(profiles, on='Customer_ID', how='left')
num_cols = df_feat.select_dtypes(exclude=['category']).columns
df_feat[num_cols] = df_feat[num_cols].fillna(0)

val_df = df_feat[df_feat['Month_Val'] > 6].copy().sort_values(['Customer_ID','Month_Val'])
val_df['pred'] = np.nan
for month in [7, 8]:
    if month > 7:
        prev = val_df[val_df['Month_Val']==month-1][['Customer_ID','pred']]
        pred_map = dict(zip(prev['Customer_ID'], prev['pred']))
        idx = val_df.index[val_df['Month_Val']==month]
        val_df.loc[idx,'target_lag1'] = val_df.loc[idx,'Customer_ID'].map(pred_map)
        val_df['target_lag1'] = val_df['target_lag1'].astype('category')
    curr = val_df[val_df['Month_Val']==month]
    probs = model.predict(curr[features_list])
    preds = np.argmax(probs, axis=1)
    val_df.loc[curr.index,'pred'] = preds

y_true = val_df['target'].astype(int)
y_pred = val_df['pred'].astype(int)
print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
print("Macro F1:", round(f1_score(y_true, y_pred, average='macro'), 4))
print(classification_report(y_true, y_pred, target_names=['Poor','Standard','Good']))
