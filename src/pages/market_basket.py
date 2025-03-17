import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
from datetime import datetime, timedelta
import numpy as np
import matplotlib.dates as mdates
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from transformers import pipeline
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

def render():
    st.title("Market Basket Analysis")
    
    # Create tabs for Apriori and FP-Growth
    tab1, tab2 = st.tabs(["Apriori Analysis", "FP-Growth Analysis"])
    
    with tab1:
        st.header("Apriori Analysis")
        # Sample dataset for Apriori
        apriori_dataset = [
            ['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
            ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
            ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
            ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
            ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']
        ]

        
        te = TransactionEncoder()
        te_ary = te.fit(apriori_dataset).transform(apriori_dataset)
        apriori_df = pd.DataFrame(te_ary, columns=te.columns_)

        # Apply Apriori algorithm
        frequent_itemsets_apriori = apriori(apriori_df, min_support=0.2, use_colnames=True)

        # Sort by support and get the top 10 itemsets
        top_itemsets_apriori = frequent_itemsets_apriori.sort_values(by="support", ascending=False).head(10)

        # Plot top 10 frequent itemsets by support
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.barplot(
            x=top_itemsets_apriori['support'],
            y=top_itemsets_apriori['itemsets'].apply(lambda x: ', '.join(x)),
            palette="viridis",
            ax=ax1
        )
        ax1.set_title('Top 10 Frequent Itemsets by Support')
        ax1.set_xlabel('Support')
        ax1.set_ylabel('Itemsets')
        st.pyplot(fig1)

        # Create a heatmap to show co-occurrence of items
        st.subheader("Co-occurrence Heatmap")
        pair_itemsets = frequent_itemsets_apriori[frequent_itemsets_apriori['itemsets'].apply(lambda x: len(x) == 2)]
        co_occurrence_matrix = pd.DataFrame(0, index=te.columns_, columns=te.columns_)
        for itemset, support in zip(pair_itemsets['itemsets'], pair_itemsets['support']):
            items = list(itemset)
            co_occurrence_matrix.loc[items[0], items[1]] = support
            co_occurrence_matrix.loc[items[1], items[0]] = support

        fig2, ax2 = plt.subplots(figsize=(12, 8))
        sns.heatmap(co_occurrence_matrix, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax2)
        ax2.set_title('Co-occurrence Heatmap of Frequent Itemsets')
        st.pyplot(fig2)

        # After existing co-occurrence heatmap in tab1 (Apriori Analysis)
        st.subheader("Purchase Time Analysis")
        time_data = pd.DataFrame({
            'Hour': np.random.randint(0, 24, 1000),
            'Sales': np.random.normal(100, 20, 1000)
        })
        
        fig_time = plt.figure(figsize=(10, 6))
        sns.histplot(data=time_data, x='Hour', weights='Sales', bins=24)
        plt.title('Sales Distribution by Hour of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Total Sales')
        st.pyplot(fig_time)

    with tab2:
        st.header("FP-Growth Analysis")
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

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

            # Remove postage items
            df = df[~df['Description'].str.contains('postage', case=False, na=False)]

            # FP-Growth Analysis
            basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)
            basket = basket > 0  

            frequent_itemsets_fpgrowth = fpgrowth(basket, min_support=0.02, use_colnames=True)

            # Sort frequent itemsets by support
            top_itemsets_fpgrowth = frequent_itemsets_fpgrowth.sort_values(by='support', ascending=False).head(10)
            top_itemsets_fpgrowth['item_names'] = top_itemsets_fpgrowth['itemsets'].apply(lambda x: ', '.join(list(x)))

            # Plot top 10 frequent itemsets by support
            st.subheader("Top 10 Frequent Itemsets (FP-Growth)")
            fig3, ax3 = plt.subplots(figsize=(12, 8))
            sns.barplot(
                x=top_itemsets_fpgrowth['support'],
                y=top_itemsets_fpgrowth['item_names'],
                palette="viridis",
                ax=ax3
            )
            ax3.set_title('Top 10 Frequent Itemsets by Support (FP-Growth)')
            ax3.set_xlabel('Support')
            ax3.set_ylabel('Itemsets')
            st.pyplot(fig3)

            # Generate association rules
            rules = association_rules(frequent_itemsets_fpgrowth, metric="confidence", min_threshold=0.5, num_itemsets=100)
            
            # Sort rules by lift
            rules = rules.sort_values('lift', ascending=False)

            # Display top 10 rules
            st.subheader("Top Association Rules")
            rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
            
            # Format and display the rules
            rules_display = rules.head(10)[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
            st.dataframe(rules_display.style.format({
                'support': '{:.3f}',
                'confidence': '{:.3f}',
                'lift': '{:.3f}'
            }))

            # Visualization of rules
            st.subheader("Association Rules Visualization")
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            scatter = ax4.scatter(rules['support'], rules['confidence'], 
                                c=rules['lift'], cmap='viridis')
            plt.colorbar(scatter, label='Lift')
            ax4.set_xlabel('Support')
            ax4.set_ylabel('Confidence')
            ax4.set_title('Support vs Confidence (color = Lift)')
            st.pyplot(fig4)

            # Add AI-Powered Product Recommendations
            st.subheader("AI-Powered Product Recommendations")
            
            # Create product embeddings
            product_descriptions = df['Description'].unique()
            le = LabelEncoder()
            encoded_products = le.fit_transform(product_descriptions)
            
            # Create product co-occurrence matrix
            n_products = len(product_descriptions)
            cooc_matrix = np.zeros((n_products, n_products))
            
            transactions = df.groupby('InvoiceNo')['Description'].agg(list)
            for transaction in transactions:
                encoded_transaction = le.transform(transaction)
                for i in encoded_transaction:
                    for j in encoded_transaction:
                        if i != j:
                            cooc_matrix[i][j] += 1
            
            # Apply PCA for visualization
            pca = PCA(n_components=2)
            product_embeddings = pca.fit_transform(cooc_matrix)
            
            # Visualize product embeddings
            st.subheader("Product Embedding Visualization")
            fig_embed = plt.figure(figsize=(12, 8))
            plt.scatter(product_embeddings[:, 0], product_embeddings[:, 1], alpha=0.5)
            plt.title("Product Embeddings")
            st.pyplot(fig_embed)
            
            # Product recommendation system
            st.subheader("Smart Product Recommendations")
            selected_product = st.selectbox("Select a product:", product_descriptions)
            
            if selected_product:
                product_idx = le.transform([selected_product])[0]
                similarities = cooc_matrix[product_idx]
                most_similar_idx = np.argsort(similarities)[-6:-1]
                recommended_products = le.inverse_transform(most_similar_idx)
                
                st.write("Recommended products:")
                for prod in recommended_products:
                    st.write(f"- {prod}")