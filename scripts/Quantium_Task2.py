import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === Load dataset ===
qvi_df = pd.read_csv("/Users/afzalnaimov/Desktop/QVI_data.csv")
qvi_df['DATE'] = pd.to_datetime(qvi_df['DATE'])
qvi_df['MONTH'] = qvi_df['DATE'].dt.to_period('M').dt.to_timestamp()

# === Monthly metrics by store ===
monthly_metrics = qvi_df.groupby(['STORE_NBR', 'MONTH']).agg(
    total_sales=('TOT_SALES', 'sum'),
    total_customers=('LYLTY_CARD_NBR', pd.Series.nunique),
    transaction_count=('TXN_ID', 'nunique')
).reset_index()
monthly_metrics['avg_txn_per_customer'] = (
    monthly_metrics['transaction_count'] / monthly_metrics['total_customers']
)

# === Defining trial and pre-trial periods ===
pre_trial_start = pd.to_datetime('2018-07-01')
pre_trial_end = pd.to_datetime('2019-01-31')
trial_start = pd.to_datetime('2019-02-01')
trial_end = pd.to_datetime('2019-04-30')

# === Correlation comparison function using pre-trial data only ===
def compare_stores(trial_store, metric_df, metric_cols):
    metric_df = metric_df[(metric_df['MONTH'] >= pre_trial_start) & (metric_df['MONTH'] <= pre_trial_end)]
    trial_data = metric_df[metric_df['STORE_NBR'] == trial_store]
    similarity_scores = []

    for store in metric_df['STORE_NBR'].unique():
        if store == trial_store:
            continue

        store_data = metric_df[metric_df['STORE_NBR'] == store]
        merged = pd.merge(trial_data, store_data, on='MONTH', suffixes=('_trial', '_control'))

        if len(merged) == 0:
            continue

        correlations = []
        for col in metric_cols:
            subset = merged[[f"{col}_trial", f"{col}_control"]].dropna()
            if len(subset) > 1:
                corr = subset.corr().iloc[0, 1]
                correlations.append(corr)

        if correlations:
            avg_corr = sum(correlations) / len(correlations)
            similarity_scores.append((store, avg_corr))

    return sorted(similarity_scores, key=lambda x: x[1], reverse=True)

# === Pick best match avoiding invalid stores ===
def pick_best_match(matches, exclude=[]):
    for store, _ in matches:
        if store not in exclude:
            return store

# === Run comparisons ===
metrics_to_compare = ['total_sales', 'total_customers']
store_77_matches = compare_stores(77, monthly_metrics, metrics_to_compare)
store_86_matches = compare_stores(86, monthly_metrics, metrics_to_compare)
store_88_matches = compare_stores(88, monthly_metrics, metrics_to_compare)

control_store_map = {
    77: pick_best_match(store_77_matches, exclude=[11]),
    86: pick_best_match(store_86_matches),
    88: pick_best_match(store_88_matches, exclude=[11])
}

# === Filter and label data ===
def get_filtered_data(trial_store, control_store, metrics_df):
    filtered = metrics_df[metrics_df['STORE_NBR'].isin([trial_store, control_store])].copy()
    filtered = filtered[(filtered['MONTH'] >= pre_trial_start) & (filtered['MONTH'] <= trial_end)]
    filtered['Period'] = filtered['MONTH'].apply(
        lambda d: 'Pre-Trial' if d < trial_start else 'Trial'
    )
    return filtered

# === Summarize % change ===
def summarize_change(data, metric):
    summary = data.groupby(['STORE_NBR', 'Period'])[metric].mean().unstack()
    summary['Percent Change'] = ((summary['Trial'] - summary['Pre-Trial']) / summary['Pre-Trial']) * 100
    return summary

# === Run analysis and summaries for all trial stores ===
all_summaries = {}

for trial_store, control_store in control_store_map.items():
    filtered_data = get_filtered_data(trial_store, control_store, monthly_metrics)

    # Summaries
    sales = summarize_change(filtered_data, 'total_sales')
    customers = summarize_change(filtered_data, 'total_customers')
    freq = summarize_change(filtered_data, 'avg_txn_per_customer')

    all_summaries[f"Store {trial_store} vs {control_store} - Sales"] = sales
    all_summaries[f"Store {trial_store} vs {control_store} - Customers"] = customers
    all_summaries[f"Store {trial_store} vs {control_store} - Frequency"] = freq

# === Final summary output ===
final_summary = pd.concat(all_summaries)
print(final_summary)

# === Prepare combined dataset for plotting ===
store_pairs = [(trial, control) for trial, control in control_store_map.items()]
all_stores = [store for pair in store_pairs for store in pair]

filtered_data = monthly_metrics[
    (monthly_metrics['STORE_NBR'].isin(all_stores)) &
    (monthly_metrics['MONTH'] >= pre_trial_start) &
    (monthly_metrics['MONTH'] <= trial_end)
].copy()

# Label stores for plotting
def label_store_group(store_id):
    for trial, control in control_store_map.items():
        if store_id == trial:
            return f"Trial {trial}"
        elif store_id == control:
            return f"Control {control}"
    return str(store_id)

filtered_data['Store Group'] = filtered_data['STORE_NBR'].apply(label_store_group)

# === Combined plots for all metrics ===
# Total Customers
plt.figure(figsize=(12, 6))
sns.lineplot(data=filtered_data, x='MONTH', y='total_customers', hue='Store Group')
plt.title("Total Customers - Trial vs Control Stores")
plt.ylabel("Total Customers")
plt.xlabel("Month")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Total Sales
plt.figure(figsize=(12, 6))
sns.lineplot(data=filtered_data, x='MONTH', y='total_sales', hue='Store Group')
plt.title("Total Sales - Trial vs Control Stores")
plt.ylabel("Total Sales")
plt.xlabel("Month")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Average Transactions per Customer
plt.figure(figsize=(12, 6))
sns.lineplot(data=filtered_data, x='MONTH', y='avg_txn_per_customer', hue='Store Group')
plt.title("Average Transactions per Customer - Trial vs Control Stores")
plt.ylabel("Avg Transactions per Customer")
plt.xlabel("Month")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()