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


st.set_page_config(page_title="Shoplytics", layout="wide")

# Add custom CSS to fix scrolling background and table styling
st.markdown("""
    <style>
        .stApp {
            background-color: transparent;
        }
        .dataframe {
            border: none !important;
        }
        .dataframe td, .dataframe th {
            border: none !important;
            background-color: transparent !important;
        }
        div[data-testid="stHorizontalBlock"] {
            background-color: transparent !important;
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("Navigation")
section = st.sidebar.selectbox("Choose Analysis", 
    options=["Market Basket Analysis", "Customer Segmentation", "Sales Forecasting"])

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

def create_lstm_model(sequence_length, n_features):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(sequence_length, n_features), return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def prepare_sequence_data(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

if section == "Market Basket Analysis":
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

elif section == "Customer Segmentation":
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

# Streamlit UI Section
elif section == "Sales Forecasting":
    st.title("Sales Forecasting")
    
    # Sample dataset for XGBoost (you can replace this with file upload)
    sample_dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Create sample sales data
    sample_data = pd.DataFrame({
        'Date': sample_dates,
        'Sales': np.random.normal(1000, 200, len(sample_dates)) + \
                np.sin(np.arange(len(sample_dates)) * 2 * np.pi / 30) * 100  # Adding seasonality
    })
    
    # Display raw data
    st.subheader("Sample Sales Data")
    st.dataframe(sample_data.head())
    
    # Feature Engineering Function
    def create_features(df):
        df = df.copy()
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        
        return df
    
    # Prepare the data
    data = create_features(sample_data)
    
    # Create lag features
    for lag in [1, 7, 14, 30]:
        data[f'sales_lag_{lag}'] = data['Sales'].shift(lag)
    
    # Create rolling mean features
    for window in [7, 14, 30]:
        data[f'sales_rolling_mean_{window}'] = data['Sales'].rolling(window=window).mean()
    
    # Drop NaN values
    data = data.dropna()
    
    # Split features and target
    feature_cols = ['Year', 'Month', 'Day', 'DayOfWeek', 'WeekOfYear'] + \
                  [col for col in data.columns if 'lag' in col or 'rolling' in col]
    
    X = data[feature_cols]
    y = data['Sales']
    
    # Train-test split
    train_size = int(len(data) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train XGBoost model
    st.subheader("Model Training")
    with st.spinner("Training XGBoost model..."):
        xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        xgb_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = xgb_model.predict(X_test)
    
    # Calculate metrics
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred))
    
    # Display metrics
    st.subheader("Model Performance Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("MSE", f"{mse:.2f}")
    col2.metric("RMSE", f"{rmse:.2f}")
    col3.metric("MAE", f"{mae:.2f}")
    
    # Plot actual vs predicted values
    st.subheader("Actual vs Predicted Sales")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    test_dates = data['Date'][train_size:].reset_index(drop=True)
    
    plt.plot(test_dates, y_test, label='Actual', alpha=0.7)
    plt.plot(test_dates, y_pred, label='Predicted', alpha=0.7)
    plt.title('Sales Forecasting: Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig1)
    
    # Feature importance plot
    st.subheader("Feature Importance Analysis")
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': xgb_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(10), x='Importance', y='Feature', palette="viridis")
    plt.title('Top 10 Most Important Features')
    plt.tight_layout()
    st.pyplot(fig2)
    
    # Sales Forecast
    st.subheader("Future Sales Forecast")
    days_to_forecast = st.slider("Select number of days to forecast", 7, 30, 7)
    
    # Generate future dates
    last_date = data['Date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_to_forecast, freq='D')
    
    # Initialize future sales predictions
    future_sales = []
    
    # Start with last known data
    last_known_sales = data[['Date', 'Sales']].set_index('Date').to_dict()['Sales']
    
    for date in future_dates:
        future_entry = {'Date': date}
        
        # Create features
        future_entry['Year'] = date.year
        future_entry['Month'] = date.month
        future_entry['Day'] = date.day
        future_entry['DayOfWeek'] = date.weekday()
        future_entry['WeekOfYear'] = date.isocalendar()[1]
        
        # Lag features (using sequentially predicted values)
        for lag in [1, 7, 14, 30]:
            past_date = date - timedelta(days=lag)
            future_entry[f'sales_lag_{lag}'] = last_known_sales.get(past_date, np.nan)
        
        # Rolling means
        for window in [7, 14, 30]:
            rolling_values = [last_known_sales.get(date - timedelta(days=i), np.nan) for i in range(1, window+1)]
            rolling_values = [x for x in rolling_values if not np.isnan(x)]
            future_entry[f'sales_rolling_mean_{window}'] = np.mean(rolling_values) if rolling_values else np.nan
        
        # Convert to DataFrame
        future_df = pd.DataFrame([future_entry])
        
        # Drop NaN values before prediction
        future_df = future_df.dropna()
        
        if not future_df.empty:
            predicted_sales = xgb_model.predict(future_df[feature_cols])[0]
        else:
            predicted_sales = np.nan
        
        # Store predictions
        future_sales.append(predicted_sales)
        last_known_sales[date] = predicted_sales  # Update for next lag reference
    
    # Plot future forecast
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    
    # Plot historical data (last 30 days)
    historical_dates = data['Date'][-30:]
    historical_sales = data['Sales'][-30:]
    
    # Ensure proper date formatting
    plt.plot(historical_dates, historical_sales, 
             label='Historical', 
             color='#1f77b4',  # Blue color
             linewidth=2)
    
    # Plot forecast with a gap between historical and forecast
    plt.plot(future_dates, future_sales, 
             label='Forecast', 
             color='#ff7f0e',  # Orange color
             linewidth=2,
             linestyle='--')
    
    # Customize the plot
    plt.title('Sales Forecast', fontsize=14, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Sales', fontsize=12)
    
    # Format x-axis
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))  # Show every 5th day
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # Rotate and align the tick labels so they look better
    plt.gcf().autofmt_xdate()
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    plt.legend(loc='upper left', frameon=True)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(fig3)
    
    # Display forecast values in a table
    # Create confidence intervals (95%)
    confidence = 0.95
    predictions_std = np.std(historical_sales)
    z_score = norm.ppf((1 + confidence) / 2)
    margin_of_error = z_score * predictions_std

    future_predictions = np.array(future_sales)
    future_predictions_with_intervals = np.column_stack([future_predictions, future_predictions - margin_of_error, future_predictions + margin_of_error])

    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecasted Sales': future_predictions_with_intervals[:, 0],
        'Lower Bound': future_predictions_with_intervals[:, 1],
        'Upper Bound': future_predictions_with_intervals[:, 2]
    })

    # Display formatted forecast table
    st.subheader("Forecasted Sales Values")
    st.dataframe(forecast_df.style.format({
        'Forecasted Sales': '{:.2f}',
        'Lower Bound': '{:.2f}',
        'Upper Bound': '{:.2f}'
    }).set_properties(**{
        'background-color': '#f0f2f6',
        'color': 'black',
        'border-color': 'white'
    }))

    # Add explanation
    st.info("""
    **Forecast Interpretation:**
    - The forecast shows expected sales for the next {} days
    - Values include 95% confidence intervals
    - Predictions are based on historical patterns and seasonality
    - Consider external factors that might affect these predictions
    """.format(days_to_forecast))

    # Add seasonality analysis
    st.subheader("Seasonality Analysis")
    seasonal_data = data.groupby('Month')['Sales'].mean().reset_index()
    
    fig_seasonal = plt.figure(figsize=(10, 6))
    sns.lineplot(data=seasonal_data, x='Month', y='Sales', marker='o')
    plt.title('Average Sales by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Sales')
    st.pyplot(fig_seasonal)

    # Add day of week analysis
    st.subheader("Day of Week Patterns")
    dow_data = data.groupby('DayOfWeek')['Sales'].mean().reset_index()
    dow_data['Day'] = dow_data['DayOfWeek'].map({
        0: 'Monday', 1: 'Tuesday', 2: 'Wednesday',
        3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
    })
    
    fig_dow = plt.figure(figsize=(10, 6))
    sns.barplot(data=dow_data, x='Day', y='Sales')
    plt.title('Average Sales by Day of Week')
    plt.xticks(rotation=45)
    st.pyplot(fig_dow)

    # Add sales distribution analysis
    st.subheader("Sales Distribution Analysis")
    fig_dist = plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='Sales', bins=50)
    plt.title('Sales Distribution')
    plt.xlabel('Sales Amount')
    plt.ylabel('Frequency')
    st.pyplot(fig_dist)

    # Add rolling statistics
    st.subheader("Rolling Statistics")
    data['7_day_avg'] = data['Sales'].rolling(window=7).mean()
    data['30_day_avg'] = data['Sales'].rolling(window=30).mean()
    
    fig_rolling = plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['Sales'], label='Daily Sales', alpha=0.5)
    plt.plot(data['Date'], data['7_day_avg'], label='7-day Moving Average')
    plt.plot(data['Date'], data['30_day_avg'], label='30-day Moving Average')
    plt.title('Sales Trends with Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig_rolling)

    # LSTM-based sales forecasting
    st.subheader("Deep Learning Sales Forecast")

    # Prepare data for LSTM
    sequence_length = 30
    n_features = 1
    scaled_sales = StandardScaler().fit_transform(data[['Sales']])
    X_lstm, y_lstm = prepare_sequence_data(scaled_sales, sequence_length)
    X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], n_features))

    # Split data
    train_size = int(len(X_lstm) * 0.8)
    X_train_lstm = X_lstm[:train_size]
    y_train_lstm = y_lstm[:train_size]
    X_test_lstm = X_lstm[train_size:]
    y_test_lstm = y_lstm[train_size:]

    # Create and train LSTM model
    lstm_model = create_lstm_model(sequence_length, n_features)
    lstm_model.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=32, verbose=0)

    # Make predictions
    lstm_predictions = lstm_model.predict(X_test_lstm)

    # Plot LSTM predictions
    st.subheader("LSTM Model Predictions")
    fig_lstm = plt.figure(figsize=(12, 6))
    plt.plot(y_test_lstm, label='Actual', alpha=0.5)
    plt.plot(lstm_predictions, label='LSTM Predictions', alpha=0.5)
    plt.title('LSTM Sales Predictions vs Actual')
    plt.legend()
    st.pyplot(fig_lstm)

    # Add anomaly detection
    st.subheader("AI-Powered Anomaly Detection")
    threshold = 2
    scaled_residuals = np.abs(y_test_lstm - lstm_predictions.flatten())
    anomalies = scaled_residuals > threshold

    fig_anomaly = plt.figure(figsize=(12, 6))
    plt.plot(range(len(y_test_lstm)), y_test_lstm, label='Actual', alpha=0.5)
    anomaly_indices = np.where(anomalies)[0]
    plt.scatter(anomaly_indices, y_test_lstm[anomaly_indices], 
               color='red', label='Anomalies', alpha=0.7)
    plt.title('Sales Anomaly Detection')
    plt.legend()
    st.pyplot(fig_anomaly)