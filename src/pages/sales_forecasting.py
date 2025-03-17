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
from src.auth.auth_handler import AuthHandler
import re

def render():
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
    # Sequence for LSTM & LSTM Model
    def prepare_sequence_data(series, sequence_length):
        X, y = [], []
        for i in range(len(series) - sequence_length):
            X.append(series[i:i + sequence_length])
            y.append(series[i + sequence_length])
        return np.array(X), np.array(y)
    
    def create_lstm_model(sequence_length, n_features):
        model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, input_shape=(sequence_length, n_features)),
         Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
    ])
        model.compile(optimizer='adam', loss='mse')
        return model


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

def is_strong_password(password):
    if len(password) < 12:
        return False, "Password must be at least 12 characters long"
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r"\d", password):
        return False, "Password must contain at least one number"
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Password must contain at least one special character"
    return True, "Password meets security requirements"

def render_register():
    st.title("Create Account")
    
    auth = AuthHandler()
    
    with st.form("register_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        st.markdown("""
        **Password Requirements:**
        - At least 12 characters long
        - Must contain uppercase and lowercase letters
        - Must contain numbers
        - Must contain special characters (!@#$%^&*(),.?":{}|<>)
        """)
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Register")
        
        if submitted:
            if password != confirm_password:
                st.error("Passwords do not match!")
                return
            
            is_strong, message = is_strong_password(password)
            if not is_strong:
                st.error(message)
                return
            
            success = auth.register_user(username, password, email)
            
            if success:
                st.success("Account created successfully!")
                st.session_state.page = "login"
                st.rerun()
            else:
                st.error("Username or email already exists!")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Already have an account? Login"):
            st.session_state.page = "login"
            st.rerun()