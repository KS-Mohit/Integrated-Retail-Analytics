import pandas as pd
from datetime import datetime
from src.database.db_handler import DatabaseHandler
import streamlit as st

class DataCollector:
    def __init__(self):
        self.db = DatabaseHandler()

    def collect_sales_data(self, file):
        try:
            df = pd.read_csv(file)
            
            # Add timestamp
            df['created_at'] = datetime.now()
            
            # Save to database
            self.db.save_sales_data(df)
            
            # Log the activity
            self.db.log_activity(
                st.session_state.user_id,
                'data_upload',
                f'Uploaded sales data with {len(df)} records'
            )
            
            return True, "Data uploaded successfully"
        except Exception as e:
            return False, f"Error uploading data: {str(e)}"

    def get_sales_data(self, start_date=None, end_date=None):
        query = "SELECT * FROM sales_data"
        if start_date and end_date:
            query += f" WHERE invoice_date BETWEEN '{start_date}' AND '{end_date}'"
        
        conn = self.db.get_connection()
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df 