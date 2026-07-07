"""
Model Evaluation Report
Runs the final trained model on a held-out slice (Months 7-8) and
generates evaluation plots: confusion matrix, ROC curves, PR curves,
feature importance bar chart.
Saves everything to reports/figures/.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import re, os, joblib, warnings
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, precision_recall_curve)
import lightgbm as lgb
warnings.filterwarnings('ignore')

os.makedirs('reports/figures', exist_ok=True)

# ── Palette / style ───────────────────────────────────────────────────────────
PALETTE = {'Poor':'#e74c3c','Standard':'#f39c12','Good':'#27ae60'}
DARK_BG='#1a1a2e'; CARD_BG='#16213e'; TEXT_COL='#eaeaea'
CS_ORDER  = ['Poor','Standard','Good']
COLORS_3  = [PALETTE[c] for c in CS_ORDER]

plt.rcParams.update({
    'figure.facecolor':DARK_BG,'axes.facecolor':CARD_BG,
    'axes.edgecolor':'#444','axes.labelcolor':TEXT_COL,
    'xtick.color':TEXT_COL,'ytick.color':TEXT_COL,
    'text.color':TEXT_COL,'font.family':'DejaVu Sans',
    'grid.color':'#333','grid.linestyle':'--','grid.linewidth':0.5,
})

def savefig(name):
    path = f'reports/figures/{name}.png'
    plt.savefig(path, dpi=140, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f'  Saved → {path}')

# ── Preprocessing helpers (same as src/preprocessing.py) ─────────────────────
from src.preprocessing import preprocess_data
from src.features import compute_lags_and_diffs, PROFILE_COLS

# ── Load artifacts ────────────────────────────────────────────────────────────
print('Loading model & artifacts …')
model      = joblib.load('model_artifacts/lightgbm_model.joblib')
profiles   = pd.read_csv('model_artifacts/customer_profiles.csv')
features_list, cat_features = joblib.load('model_artifacts/features_info.joblib')

# ── Prepare validation data (Months 7 & 8) ───────────────────────────────────
print('Preprocessing training data for validation …')
train_raw = pd.read_csv('Data/train.csv', low_memory=False)
df_clean  = preprocess_data(train_raw)
df_lags   = compute_lags_and_diffs(df_clean)

target_map  = {'Poor':0,'Standard':1,'Good':2}
inv_tgt_map = {0:'Poor',1:'Standard',2:'Good'}
df_lags['target']     = df_lags['Credit_Score'].map(target_map)
df_lags['target_lag1']= df_lags.groupby('Customer_ID')['target'].shift(1).astype('category')

# Merge profiles
df_feat = df_lags.merge(profiles, on='Customer_ID', how='left')
num_cols = df_feat.select_dtypes(exclude=['category']).columns
df_feat[num_cols] = df_feat[num_cols].fillna(0)

val_df = df_feat[df_feat['Month_Val'] > 6].copy().sort_values(['Customer_ID','Month_Val'])

# Auto-regressive sequential prediction on val (July → August)
val_df['pred'] = np.nan
all_indices = []
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
    val_df.loc[curr.index,'pred']   = preds
    val_df.loc[curr.index,'probs_Poor']     = probs[:,0]
    val_df.loc[curr.index,'probs_Standard'] = probs[:,1]
    val_df.loc[curr.index,'probs_Good']     = probs[:,2]
    all_indices.extend(curr.index.tolist())

result = val_df.loc[all_indices].copy()
y_true  = result['target'].astype(int)
y_pred  = result['pred'].astype(int)
y_probs = result[['probs_Poor','probs_Standard','probs_Good']].values

print(f'Validation rows: {len(result)}')

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 10  –  Confusion matrix  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ═══════════════════════════════════════════════════════════════════════════════
print('\n[10] Confusion matrix …')
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, data, title, fmt in zip(axes,
    [cm, cm_norm*100],
    ['Confusion Matrix (counts)', 'Confusion Matrix (row %)'],
    ['d', '.1f']):
    cmap = sns.color_palette('crest', as_cmap=True)
    sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=CS_ORDER, yticklabels=CS_ORDER,
                linewidths=1, linecolor=DARK_BG,
                annot_kws={'size':13,'fontweight':'bold'},
                ax=ax, cbar_kws={'shrink':0.8})
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', color=TEXT_COL, pad=10)
    ax.tick_params(labelsize=10)
fig.suptitle(f'Model Validation — Auto-Regressive Sequential Prediction',
             fontsize=13, fontweight='bold', color=TEXT_COL, y=1.02)
plt.tight_layout()
savefig('10_confusion_matrix')

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 11  –  ROC curves (one-vs-rest)  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ═══════════════════════════════════════════════════════════════════════════════
print('[11] ROC curves …')
fig, ax = plt.subplots(figsize=(7, 5.5))
for i, (cs, col) in enumerate(zip(CS_ORDER, COLORS_3)):
    y_bin = (y_true == i).astype(int)
    fpr, tpr, _ = roc_curve(y_bin, y_probs[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=col, lw=2.2, label=f'{cs} (AUC = {roc_auc:.3f})')
ax.plot([0,1],[0,1], 'w--', lw=1, alpha=0.4)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — One vs. Rest', fontsize=13, fontweight='bold', color=TEXT_COL)
ax.legend(facecolor=CARD_BG, edgecolor='#555', labelcolor=TEXT_COL)
ax.grid(True, alpha=0.3)
savefig('11_roc_curves')

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 12  –  Precision-Recall curves  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ═══════════════════════════════════════════════════════════════════════════════
print('[12] Precision-Recall curves …')
fig, ax = plt.subplots(figsize=(7, 5.5))
for i, (cs, col) in enumerate(zip(CS_ORDER, COLORS_3)):
    y_bin = (y_true == i).astype(int)
    prec, rec, _ = precision_recall_curve(y_bin, y_probs[:, i])
    pr_auc = auc(rec, prec)
    ax.plot(rec, prec, color=col, lw=2.2, label=f'{cs} (AUC = {pr_auc:.3f})')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves — One vs. Rest', fontsize=13, fontweight='bold', color=TEXT_COL)
ax.legend(facecolor=CARD_BG, edgecolor='#555', labelcolor=TEXT_COL)
ax.grid(True, alpha=0.3)
savefig('12_precision_recall')

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 13  –  Feature importance  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ═══════════════════════════════════════════════════════════════════════════════
print('[13] Feature importance …')
imp = pd.DataFrame({
    'feature': features_list,
    'importance': model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False).head(30)

fig, ax = plt.subplots(figsize=(9, 9))
colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(imp)))
bars = ax.barh(imp['feature'][::-1], imp['importance'][::-1],
               color=colors[::-1], edgecolor='none')
ax.set_xlabel('Feature Gain (LightGBM)')
ax.set_title('Top-30 Feature Importance (Gain)', fontsize=13, fontweight='bold', color=TEXT_COL)
ax.grid(True, axis='x', alpha=0.3)
# Add value labels
for bar, val in zip(bars, imp['importance'][::-1]):
    ax.text(bar.get_width()*1.005, bar.get_y()+bar.get_height()/2,
            f'{val:,.0f}', va='center', fontsize=7, color=TEXT_COL)
savefig('13_feature_importance')

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 14  –  Class-wise recall by month  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ═══════════════════════════════════════════════════════════════════════════════
print('[14] Per-month metrics …')
months_data = []
for month in [7, 8]:
    sub = result[result['Month_Val'] == month]
    yt  = sub['target'].astype(int)
    yp  = sub['pred'].astype(int)
    cm_m = confusion_matrix(yt, yp, labels=[0,1,2])
    recalls = cm_m.diagonal() / cm_m.sum(axis=1).clip(min=1)
    for ci, cs in enumerate(CS_ORDER):
        months_data.append({'Month': f'Month {month}', 'Credit_Score': cs, 'Recall': recalls[ci]})

mdf = pd.DataFrame(months_data)
fig, ax = plt.subplots(figsize=(7, 4.5))
width = 0.25
x = np.arange(2)
for i, (cs, col) in enumerate(zip(CS_ORDER, COLORS_3)):
    sub = mdf[mdf['Credit_Score'] == cs]
    ax.bar(x + i*width, sub['Recall']*100, width=width,
           color=col, label=cs, edgecolor=DARK_BG, linewidth=0.5)
ax.set_xticks(x + width)
ax.set_xticklabels(['July (Month 7)', 'August (Month 8)'])
ax.set_ylabel('Recall (%)'); ax.set_ylim(0, 100)
ax.set_title('Per-class Recall by Validation Month', fontsize=13, fontweight='bold', color=TEXT_COL)
ax.legend(facecolor=CARD_BG, edgecolor='#555', labelcolor=TEXT_COL)
ax.grid(True, axis='y', alpha=0.3)
savefig('14_monthly_recall')

# ═══════════════════════════════════════════════════════════════════════════════
# Print final metrics summary
# ═══════════════════════════════════════════════════════════════════════════════
from sklearn.metrics import accuracy_score, f1_score
acc   = accuracy_score(y_true, y_pred)
macro = f1_score(y_true, y_pred, average='macro')
print('\n' + '='*55)
print(f'  FINAL VALIDATION RESULTS  (Months 7-8, n={len(result):,})')
print('='*55)
print(f'  Accuracy  : {acc:.4f}')
print(f'  Macro F1  : {macro:.4f}')
print('\n' + classification_report(y_true, y_pred, target_names=CS_ORDER))
print('✓ All evaluation figures saved to reports/figures/')
