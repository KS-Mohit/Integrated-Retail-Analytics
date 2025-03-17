import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

dataset = [
    ['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
    ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
    ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
    ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
    ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']
]

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)

top_itemsets = frequent_itemsets.sort_values(by="support", ascending=False).head(10)

# Top 10 frequent itemsets by support
plt.figure(figsize=(10, 6))
sns.barplot(
    x=top_itemsets['support'],
    y=top_itemsets['itemsets'].apply(lambda x: ', '.join(x)),
    palette="viridis"
)
plt.title('Top 10 Frequent Itemsets by Support')
plt.xlabel('Support')
plt.ylabel('Itemsets')
plt.show()

# Heatmap
pair_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) == 2)]
co_occurrence_matrix = pd.DataFrame(0, index=te.columns_, columns=te.columns_)

for itemset, support in zip(pair_itemsets['itemsets'], pair_itemsets['support']):
    items = list(itemset)
    co_occurrence_matrix.loc[items[0], items[1]] = support
    co_occurrence_matrix.loc[items[1], items[0]] = support

plt.figure(figsize=(12, 8))
sns.heatmap(co_occurrence_matrix, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title('Co-occurrence Heatmap of Frequent Itemsets')
plt.show()
