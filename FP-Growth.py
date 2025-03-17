import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('OnlineRetailShopGermany.csv')

# Data cleaning
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
df = df.drop_duplicates()
df = df[df['Quantity'] > 0]
df = df[df['Quantity'] < 1000]
df = df[df['UnitPrice'] > 0]
df['Quantity'] = df['Quantity'].fillna(df['Quantity'].median())
df['UnitPrice'] = df['UnitPrice'].fillna(df['UnitPrice'].median())
df['Description'] = df['Description'].fillna(df['Description'].mode()[0])
df['Country'] = df['Country'].str.strip().str.lower()
df['InvoiceMonth'] = df['InvoiceDate'].dt.month
df['InvoiceYear'] = df['InvoiceDate'].dt.year
df['InvoiceDayOfWeek'] = df['InvoiceDate'].dt.dayofweek
df = df[~df['Description'].str.contains('postage', case=False, na=False)]

# FP-Growth
basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)
basket = basket > 0  

frequent_itemsets = fpgrowth(basket, min_support=0.02, use_colnames=True)
rules = association_rules(frequent_itemsets, num_itemsets=94, metric="lift", min_threshold=1.0, support_only=False)

print("Frequent Itemsets:\n", frequent_itemsets.head())
print("\nAssociation Rules:\n", rules.head())


frequent_itemsets_sorted = frequent_itemsets.sort_values(by='support', ascending=False)
top_10_itemsets = frequent_itemsets_sorted.head(10)
top_10_itemsets['item_names'] = top_10_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))


plt.figure(figsize=(5, 1))  
sns.barplot(
    x=top_10_itemsets['support'],
    y=top_10_itemsets['item_names'],
    palette="viridis"
)
plt.title('Top 10 Frequent Itemsets by Support', fontsize=10) 
plt.xlabel('Support', fontsize=8)
plt.ylabel('Itemsets', fontsize=8)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)

plt.subplots_adjust(left=0.4)

plt.show()

# RFM analysis
if df['CustomerID'].isnull().any():
    print("Warning: Some CustomerIDs are missing. Removing rows with missing CustomerIDs.")
    df = df.dropna(subset=['CustomerID'])  

print("Number of unique CustomerIDs:", df['CustomerID'].nunique())

df['TotalPurchase'] = df['Quantity'] * df['UnitPrice']

snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

if 'CustomerID' not in df.columns:
    raise ValueError("CustomerID column is missing in the dataset")

rfm_df = df.groupby('CustomerID').agg(
    Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
    Frequency=('InvoiceNo', 'nunique'),
    Monetary=('TotalPurchase', 'sum')
).reset_index()

scaler = StandardScaler()
rfm_df_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])

kmeans = KMeans(n_clusters=4, random_state=42)
rfm_df['Cluster'] = kmeans.fit_predict(rfm_df_scaled)

print(rfm_df)
