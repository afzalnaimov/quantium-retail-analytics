# Quantium Virtual Internship - Task One
# Author: Afzal Naimov

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Load the data ===
purchase_df = pd.read_csv("/Users/afzalnaimov/Desktop/QVI_purchase_behaviour.csv")
transaction_df = pd.read_excel("/Users/afzalnaimov/Desktop/QVI_transaction_data.xlsx")

# === Convert Excel date format to datetime ===
transaction_df["DATE"] = pd.to_datetime("1899-12-30") + pd.to_timedelta(transaction_df["DATE"], unit="D")

# === Merge both datasets on loyalty card number ===
merged_df = transaction_df.merge(purchase_df, on="LYLTY_CARD_NBR", how="left")

# === Extract pack size from product name ===
merged_df["PACK_SIZE"] = merged_df["PROD_NAME"].str.extract(r"(\d+)\s?g", expand=False).astype(float)

# === Extract brand from product name (first word) ===
merged_df["BRAND"] = merged_df["PROD_NAME"].str.split().str[0]

# === Remove quantity outliers (e.g. people buying 200+ bags at once) ===
merged_df = merged_df[merged_df["PROD_QTY"] < 200]

# === Group by customer segment ===
segment_summary = merged_df.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).agg({
    'TOT_SALES': 'sum',
    'TXN_ID': 'nunique',
    'PROD_QTY': 'sum'
}).reset_index()

# === Calculate average spend per transaction ===
segment_summary['Avg_Spend_per_Transaction'] = segment_summary['TOT_SALES'] / segment_summary['TXN_ID']

# === Display the results ===
print(segment_summary)

# === In order to better understand which segments are spending the most, lets visualize it ===
sns.barplot(data=segment_summary, x='LIFESTAGE', y='TOT_SALES', hue='PREMIUM_CUSTOMER')
plt.title('Total Chip Sales by Customer Segment')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# === Key findings from Task 1 ===

# Young Families are the highest spenders on chips overall, driven by frequent and high-volume purchases — 
#especially in the Budget and Mainstream tiers.

# Premium Singles and Couples have the highest average spend per transaction, 
#indicating a preference for quality or premium brands despite fewer shopping trips.

# Budget Older Singles/Couples are the lowest spenders, both in volume and transaction value — 
#showing high price sensitivity and limited opportunity for upselling.
