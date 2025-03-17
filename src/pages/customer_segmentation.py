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
    st.title("Customer Segmentation")
    
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Data preprocessing (from FP-Growth section)
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

        # RFM Analysis
        if df['CustomerID'].isnull().any():
            df = df.dropna(subset=['CustomerID'])

        df['TotalPurchase'] = df['Quantity'] * df['UnitPrice']
        snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

        rfm_df = df.groupby('CustomerID').agg(
            Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
            Frequency=('InvoiceNo', 'nunique'),
            Monetary=('TotalPurchase', 'sum')
        ).reset_index()

        scaler = StandardScaler()
        rfm_df_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])

        # Random Forest Classifier
        rfm_df['Cluster'] = pd.qcut(rfm_df['Monetary'], 4, labels=[0, 1, 2, 3])

        X = rfm_df[['Recency', 'Frequency', 'Monetary']]
        y = rfm_df['Cluster']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, y_train)

        y_pred = rf_model.predict(X_test)

        st.header("RFM Analysis Results")
        st.write("### RFM Data with Clusters")
        st.dataframe(rfm_df)

        st.write("### Random Forest Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format({
            "precision": "{:.2f}", 
            "recall": "{:.2f}", 
            "f1-score": "{:.2f}", 
            "support": "{:.0f}"
        }))

        st.write("### Model Accuracy")
        st.write(accuracy_score(y_test, y_pred))

        # Feature importance visualization
        st.write("### Feature Importance")
        feature_importance = pd.DataFrame(
            rf_model.feature_importances_, 
            index=['Recency', 'Frequency', 'Monetary'], 
            columns=["Importance"]
        ).sort_values(by="Importance", ascending=False)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x=feature_importance.Importance, y=feature_importance.index, palette="viridis", ax=ax)
        ax.set_title("Feature Importance in Customer Segmentation")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Features")
        st.pyplot(fig)

        # Add purchase frequency analysis
        st.subheader("Purchase Frequency Distribution")
        fig_freq = plt.figure(figsize=(10, 6))
        sns.histplot(data=rfm_df, x='Frequency', bins=30)
        plt.title('Customer Purchase Frequency Distribution')
        st.pyplot(fig_freq)

        # Add customer value segments
        st.subheader("Customer Value Segments")
        rfm_df['Value_Segment'] = pd.qcut(rfm_df['Monetary'], q=4, labels=['Bronze', 'Silver', 'Gold', 'Platinum'])
        value_counts = rfm_df['Value_Segment'].value_counts()
        
        fig_segments = plt.figure(figsize=(10, 6))
        plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%')
        plt.title('Customer Value Segments Distribution')
        st.pyplot(fig_segments)

        # Add recency vs monetary scatter plot
        st.subheader("Recency vs Monetary Value")
        fig_scatter = plt.figure(figsize=(10, 6))
        plt.scatter(rfm_df['Recency'], rfm_df['Monetary'], alpha=0.5)
        plt.xlabel('Recency (days)')
        plt.ylabel('Monetary Value')
        st.pyplot(fig_scatter)

        # Add Advanced Customer Behavior Analysis
        st.subheader("Advanced Customer Behavior Analysis")
        
        # Prepare features for deep learning
        X_deep = np.column_stack([
            rfm_df['Recency'],
            rfm_df['Frequency'],
            rfm_df['Monetary']
        ])
        
        # Normalize features
        scaler = StandardScaler()
        X_deep_scaled = scaler.fit_transform(X_deep)
        
        # Apply t-SNE for visualization
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X_deep_scaled)
        
        # Visualize customer segments in 2D
        fig_tsne = plt.figure(figsize=(10, 8))
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=rfm_df['Cluster'], cmap='viridis')
        plt.title("Customer Segments Visualization (t-SNE)")
        plt.colorbar(label='Cluster')
        st.pyplot(fig_tsne)
        
        # Customer lifetime value prediction
        st.subheader("AI-Powered Customer Lifetime Value Prediction")
        
        # Create and train a simple neural network
        model = Sequential([
            Dense(64, activation='relu', input_shape=(3,)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_deep_scaled, rfm_df['Monetary'], epochs=50, batch_size=32, verbose=0)
        
        # Add prediction interface
        st.write("Predict Customer Lifetime Value:")
        col1, col2, col3 = st.columns(3)
        with col1:
            recency = st.number_input("Recency (days)", min_value=0)
        with col2:
            frequency = st.number_input("Frequency", min_value=0)
        with col3:
            monetary = st.number_input("Monetary Value", min_value=0.0)
        
        if st.button("Predict CLV"):
            input_data = scaler.transform([[recency, frequency, monetary]])
            prediction = model.predict(input_data)[0][0]
            st.write(f"Predicted Customer Lifetime Value: ${prediction:.2f}")