import streamlit as st

def load_css():
    """Load and apply custom CSS styles"""
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
            .custom-metric-container {
                padding: 1rem;
                border-radius: 0.5rem;
                background-color: #f8f9fa;
                margin: 0.5rem 0;
            }
            .chart-container {
                margin: 1.5rem 0;
                padding: 1rem;
                border: 1px solid #e9ecef;
                border-radius: 0.5rem;
            }
        </style>
    """, unsafe_allow_html=True)

def create_metric_container(title, value, description=""):
    """Create a styled metric container"""
    st.markdown(f"""
        <div class="custom-metric-container">
            <h4>{title}</h4>
            <h2>{value}</h2>
            <p>{description}</p>
        </div>
    """, unsafe_allow_html=True)

def create_chart_container():
    """Create a styled container for charts"""
    return st.markdown("""
        <div class="chart-container">
        </div>
    """, unsafe_allow_html=True) 