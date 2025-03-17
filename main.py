import streamlit as st
import os
from src.pages import market_basket, customer_segmentation, sales_forecasting, login, register
from src.utils.visualization import load_css
from src.auth.auth_handler import AuthHandler
from src.database.db_handler import DatabaseHandler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Ensure the data directory exists
os.makedirs('data', exist_ok=True)

def main():
    st.set_page_config(page_title="Shoplytics", layout="wide")
    load_css()

    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'page' not in st.session_state:
        st.session_state.page = 'login'

    # Authentication check
    if not st.session_state.authenticated:
        if st.session_state.page == 'register':
            register.render_register()
        else:
            login.render_login()
        return

    # Main navigation
    st.sidebar.title("Navigation")
    section = st.sidebar.selectbox("Choose Analysis", 
        options=["Market Basket Analysis", "Customer Segmentation", "Sales Forecasting"])

    # Add logout button
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.user_id = None
        st.session_state.token = None
        st.session_state.page = 'login'
        st.rerun()

    # Render selected section
    if section == "Market Basket Analysis":
        market_basket.render()
    elif section == "Customer Segmentation":
        customer_segmentation.render()
    elif section == "Sales Forecasting":
        sales_forecasting.render()

if __name__ == "__main__":
    main() 