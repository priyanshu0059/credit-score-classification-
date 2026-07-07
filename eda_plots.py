"""
EDA Plots for Credit Score Classification
Generates all Exploratory Data Analysis figures and saves to reports/figures/
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import re
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('reports/figures', exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = {'Poor': '#e74c3c', 'Standard': '#f39c12', 'Good': '#27ae60'}
COLORS_3 = ['#e74c3c', '#f39c12', '#27ae60']
DARK_BG   = '#1a1a2e'
CARD_BG   = '#16213e'
TEXT_COL  = '#eaeaea'
ACC_TEAL  = '#0f3460'
ACC_PINK  = '#e94560'

plt.rcParams.update({
    'figure.facecolor': DARK_BG, 'axes.facecolor': CARD_BG,
    'axes.edgecolor': '#444', 'axes.labelcolor': TEXT_COL,
    'xtick.color': TEXT_COL, 'ytick.color': TEXT_COL,
    'text.color': TEXT_COL, 'font.family': 'DejaVu Sans',
    'grid.color': '#333', 'grid.linestyle': '--', 'grid.linewidth': 0.5,
})

# ── Helper functions ──────────────────────────────────────────────────────────
def clean_float(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().replace('_','')
    if '__' in s: s = s.replace('__','')
    if s in ('', 'nan'): return np.nan
    try: return float(s)
    except: return np.nan

def clean_int(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().replace('_','')
    if s in ('', 'nan'): return np.nan
    try: return int(s)
    except: return np.nan

def parse_history_age(val):
    if pd.isna(val) or not isinstance(val, str): return np.nan
    m = re.match(r'(\d+)\s+Years\s+and\s+(\d+)\s+Months', val.strip())
    if m: return int(m.group(1))*12 + int(m.group(2))
    return np.nan

def savefig(name):
    path = f'reports/figures/{name}.png'
    plt.savefig(path, dpi=140, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f'  Saved → {path}')

# ── Load & quick-clean ────────────────────────────────────────────────────────
print('Loading training data …')
df = pd.read_csv('Data/train.csv', low_memory=False)

df['Age']                 = df['Age'].apply(clean_int)
df['Annual_Income']       = df['Annual_Income'].apply(clean_float)
df['Outstanding_Debt']    = df['Outstanding_Debt'].apply(clean_float)
df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].apply(clean_float)
df['Num_of_Loan']         = df['Num_of_Loan'].apply(clean_int)
df['Changed_Credit_Limit']= df['Changed_Credit_Limit'].apply(clean_float)
df['Amount_invested_monthly'] = df['Amount_invested_monthly'].apply(clean_float)
df['Monthly_Balance']     = df['Monthly_Balance'].apply(clean_float)
df['History_Age_Months']  = df['Credit_History_Age'].apply(parse_history_age)

# Cap obvious outliers
df.loc[(df['Age'] < 18) | (df['Age'] > 80), 'Age'] = np.nan
df.loc[(df['Num_of_Loan'] < 0) | (df['Num_of_Loan'] > 15), 'Num_of_Loan'] = np.nan
df.loc[df['Num_of_Delayed_Payment'] > 28, 'Num_of_Delayed_Payment'] = np.nan
df.loc[df['Monthly_Balance'] < 0, 'Monthly_Balance'] = np.nan
df['Occupation'] = df['Occupation'].replace('_______', np.nan).fillna('Unknown')
df['Credit_Mix'] = df['Credit_Mix'].replace('_', np.nan).fillna('Unknown')
df['Payment_Behaviour'] = df['Payment_Behaviour'].replace('!@9#%8', np.nan).fillna('Unknown')

CS_ORDER = ['Poor', 'Standard', 'Good']
cs_colors = [PALETTE[c] for c in CS_ORDER]

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1  –  Class distribution (donut)
# ═══════════════════════════════════════════════════════════════════════════════
print('\n[1/9] Class distribution …')
counts = df['Credit_Score'].value_counts().reindex(CS_ORDER)
fig, ax = plt.subplots(figsize=(7, 5))
wedges, texts, autotexts = ax.pie(
    counts, labels=CS_ORDER, colors=cs_colors,
    autopct='%1.1f%%', startangle=90,
    wedgeprops=dict(width=0.55, edgecolor=DARK_BG, linewidth=2),
    textprops={'color': TEXT_COL, 'fontsize': 11}
)
for at in autotexts:
    at.set_fontweight('bold'); at.set_fontsize(12)
ax.set_title('Credit Score Class Distribution', fontsize=14, fontweight='bold', pad=18, color=TEXT_COL)
for i, (w, c, n) in enumerate(zip(counts, CS_ORDER, CS_ORDER)):
    ax.text(0, -0.07 + i*0.07 - 0.07, f'{n}: {w:,}', ha='center',
            fontsize=9, color=TEXT_COL, transform=ax.transAxes)
savefig('01_class_distribution')

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2  –  Age distribution by credit score
# ═══════════════════════════════════════════════════════════════════════════════
print('[2/9] Age distribution …')
fig, ax = plt.subplots(figsize=(9, 4.5))
for cs, col in PALETTE.items():
    subset = df[df['Credit_Score'] == cs]['Age'].dropna()
    ax.hist(subset, bins=30, alpha=0.65, color=col, label=cs, edgecolor='none')
    ax.axvline(subset.median(), color=col, linestyle='--', linewidth=1.5)
ax.set_xlabel('Age'); ax.set_ylabel('Count')
ax.set_title('Age Distribution by Credit Score', fontsize=13, fontweight='bold', color=TEXT_COL)
ax.legend(facecolor=CARD_BG, edgecolor='#555', labelcolor=TEXT_COL)
ax.grid(True, alpha=0.3)
savefig('02_age_distribution')

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3  –  Annual income box-plot
# ═══════════════════════════════════════════════════════════════════════════════
print('[3/9] Annual income …')
fig, ax = plt.subplots(figsize=(8, 5))
data = [df[df['Credit_Score']==cs]['Annual_Income'].dropna() for cs in CS_ORDER]
bp = ax.boxplot(data, patch_artist=True, notch=True, vert=True,
                medianprops=dict(color='white', linewidth=2),
                whiskerprops=dict(color='#999'),
                capprops=dict(color='#999'),
                flierprops=dict(marker='o', markersize=2, alpha=0.2,
                                markerfacecolor='#aaa', markeredgecolor='none'))
for patch, col in zip(bp['boxes'], cs_colors):
    patch.set_facecolor(col); patch.set_alpha(0.65)
ax.set_xticks([1,2,3]); ax.set_xticklabels(CS_ORDER)
ax.set_ylabel('Annual Income (USD)'); ax.set_yscale('log')
ax.set_title('Annual Income by Credit Score  (log scale)', fontsize=13, fontweight='bold', color=TEXT_COL)
ax.grid(True, axis='y', alpha=0.3)
savefig('03_annual_income_boxplot')

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4  –  Number of delayed payments violin
# ═══════════════════════════════════════════════════════════════════════════════
print('[4/9] Delayed payments …')
plot_df = df[['Credit_Score','Num_of_Delayed_Payment']].dropna()
plot_df = plot_df[plot_df['Credit_Score'].isin(CS_ORDER)]

fig, ax = plt.subplots(figsize=(8, 5))
parts = ax.violinplot(
    [plot_df[plot_df['Credit_Score']==cs]['Num_of_Delayed_Payment'].values for cs in CS_ORDER],
    positions=[1,2,3], showmedians=True, showextrema=False
)
for i, (body, col) in enumerate(zip(parts['bodies'], cs_colors)):
    body.set_facecolor(col); body.set_alpha(0.6)
parts['cmedians'].set_color('white'); parts['cmedians'].set_linewidth(2)
ax.set_xticks([1,2,3]); ax.set_xticklabels(CS_ORDER)
ax.set_ylabel('# Delayed Payments')
ax.set_title('Delayed Payments Distribution by Credit Score', fontsize=13, fontweight='bold', color=TEXT_COL)
ax.grid(True, axis='y', alpha=0.3)
savefig('04_delayed_payments_violin')

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5  –  Outstanding debt vs credit utilisation scatter
# ═══════════════════════════════════════════════════════════════════════════════
print('[5/9] Debt vs utilisation …')
samp = df.sample(min(8000, len(df)), random_state=42)
fig, ax = plt.subplots(figsize=(8, 5))
for cs, col in PALETTE.items():
    sub = samp[samp['Credit_Score'] == cs]
    ax.scatter(sub['Outstanding_Debt'], sub['Credit_Utilization_Ratio'],
               c=col, alpha=0.35, s=12, label=cs, edgecolors='none')
ax.set_xlabel('Outstanding Debt (USD)'); ax.set_ylabel('Credit Utilisation Ratio (%)')
ax.set_title('Outstanding Debt vs Credit Utilisation', fontsize=13, fontweight='bold', color=TEXT_COL)
ax.legend(facecolor=CARD_BG, edgecolor='#555', labelcolor=TEXT_COL, markerscale=2)
ax.grid(True, alpha=0.3)
savefig('05_debt_vs_utilisation')

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6  –  Credit Mix bar chart
# ═══════════════════════════════════════════════════════════════════════════════
print('[6/9] Credit mix …')
cm_df = df[df['Credit_Mix'].isin(['Good','Standard','Bad'])].groupby(
    ['Credit_Mix','Credit_Score']).size().unstack(fill_value=0).reindex(columns=CS_ORDER, fill_value=0)

fig, ax = plt.subplots(figsize=(8, 5))
cm_df_pct = cm_df.div(cm_df.sum(axis=1), axis=0) * 100
bottom = np.zeros(len(cm_df_pct))
for cs, col in zip(CS_ORDER, cs_colors):
    ax.bar(cm_df_pct.index, cm_df_pct[cs], bottom=bottom, color=col,
           label=cs, edgecolor=DARK_BG, linewidth=0.5)
    bottom += cm_df_pct[cs].values
ax.set_ylabel('Percentage (%)'); ax.set_xlabel('Credit Mix')
ax.set_title('Credit Score Distribution within Credit Mix Types', fontsize=13, fontweight='bold', color=TEXT_COL)
ax.legend(facecolor=CARD_BG, edgecolor='#555', labelcolor=TEXT_COL)
ax.grid(True, axis='y', alpha=0.3)
savefig('06_credit_mix_stacked')

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7  –  Occupation-level Good-score rate
# ═══════════════════════════════════════════════════════════════════════════════
print('[7/9] Occupation analysis …')
occ_df = df[df['Occupation'] != 'Unknown'].copy()
occ_rate = (occ_df.groupby('Occupation')
            .apply(lambda x: (x['Credit_Score']=='Good').mean())
            .sort_values(ascending=False))

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(occ_rate.index, occ_rate.values*100,
               color=[plt.cm.RdYlGn(v) for v in occ_rate.values],
               edgecolor='none')
ax.set_xlabel('% with Good Credit Score')
ax.set_title('Good Credit Score Rate by Occupation', fontsize=13, fontweight='bold', color=TEXT_COL)
ax.invert_yaxis()
ax.axvline(occ_rate.mean()*100, color='white', linestyle='--', linewidth=1.2, label='Mean')
ax.legend(facecolor=CARD_BG, edgecolor='#555', labelcolor=TEXT_COL)
ax.grid(True, axis='x', alpha=0.3)
# Value labels
for bar, v in zip(bars, occ_rate.values):
    ax.text(v*100 + 0.2, bar.get_y() + bar.get_height()/2,
            f'{v*100:.1f}%', va='center', fontsize=9, color=TEXT_COL)
savefig('07_occupation_good_rate')

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 8  –  Correlation heatmap
# ═══════════════════════════════════════════════════════════════════════════════
print('[8/9] Correlation heatmap …')
num_cols = ['Age','Annual_Income','Monthly_Inhand_Salary','Num_Bank_Accounts',
            'Num_Credit_Card','Interest_Rate','Num_of_Loan',
            'Delay_from_due_date','Num_of_Delayed_Payment',
            'Outstanding_Debt','Credit_Utilization_Ratio','History_Age_Months',
            'Total_EMI_per_month','Amount_invested_monthly','Monthly_Balance']
target_num = df['Credit_Score'].map({'Poor':0,'Standard':1,'Good':2})
corr_df = df[num_cols].copy()
corr_df['Credit_Score_num'] = target_num
corr = corr_df.corr()

fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(10, 130, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.6, vmin=-0.6, center=0,
            square=True, linewidths=0.3, linecolor='#333',
            annot=True, fmt='.2f', annot_kws={'size': 7},
            cbar_kws={'shrink': 0.7}, ax=ax)
ax.set_title('Feature Correlation Matrix (lower-triangle)', fontsize=13, fontweight='bold', color=TEXT_COL, pad=14)
ax.tick_params(labelsize=8)
plt.xticks(rotation=40, ha='right'); plt.yticks(rotation=0)
savefig('08_correlation_heatmap')

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 9  –  Payment behaviour breakdown
# ═══════════════════════════════════════════════════════════════════════════════
print('[9/9] Payment behaviour …')
pb_valid = ['High_spent_Small_value_payments','High_spent_Medium_value_payments',
            'High_spent_Large_value_payments','Low_spent_Small_value_payments',
            'Low_spent_Medium_value_payments','Low_spent_Large_value_payments']
pb_df = df[df['Payment_Behaviour'].isin(pb_valid)].groupby(
    ['Payment_Behaviour','Credit_Score']).size().unstack(fill_value=0).reindex(columns=CS_ORDER, fill_value=0)
pb_pct = pb_df.div(pb_df.sum(axis=1), axis=0)*100
# Shorten labels
pb_pct.index = pb_pct.index.str.replace('_payments','').str.replace('_value','').str.replace('_spent',' spent')

fig, ax = plt.subplots(figsize=(11, 5))
bottom = np.zeros(len(pb_pct))
for cs, col in zip(CS_ORDER, cs_colors):
    ax.bar(range(len(pb_pct)), pb_pct[cs], bottom=bottom, color=col, label=cs,
           edgecolor=DARK_BG, linewidth=0.5, width=0.6)
    bottom += pb_pct[cs].values
ax.set_xticks(range(len(pb_pct)))
ax.set_xticklabels(pb_pct.index, rotation=28, ha='right', fontsize=9)
ax.set_ylabel('Percentage (%)'); ax.set_title('Credit Score Breakdown by Payment Behaviour',
                                               fontsize=13, fontweight='bold', color=TEXT_COL)
ax.legend(facecolor=CARD_BG, edgecolor='#555', labelcolor=TEXT_COL)
ax.grid(True, axis='y', alpha=0.3)
savefig('09_payment_behaviour')

print('\n✓ All 9 EDA figures saved to reports/figures/')
