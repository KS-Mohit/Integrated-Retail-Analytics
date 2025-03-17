import sqlite3
import os
from datetime import datetime
import pandas as pd

class DatabaseHandler:
    def __init__(self):
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        self.db_path = os.path.join('data', 'shoplytics.db')
        self.init_database()

    def init_database(self):
        print(f"Initializing database at: {self.db_path}")  # Debug print
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Create users table
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                role TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create sales_data table
        c.execute('''
            CREATE TABLE IF NOT EXISTS sales_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                invoice_no TEXT,
                stock_code TEXT,
                description TEXT,
                quantity INTEGER,
                invoice_date TIMESTAMP,
                unit_price REAL,
                customer_id TEXT,
                country TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create activity_log table
        c.execute('''
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                action TEXT,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')

        conn.commit()
        conn.close()
        print("Database initialized successfully")  # Debug print

    def add_user(self, username, password_hash, email, role='user'):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        try:
            c.execute(
                'INSERT INTO users (username, password, email, role) VALUES (?, ?, ?, ?)',
                (username, password_hash, email, role)
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()

    def get_user(self, username):
        print(f"Attempting to get user: {username}")  # Debug print
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = c.fetchone()
        conn.close()
        print(f"User found: {user}")  # Debug print
        return user

    def log_activity(self, user_id, action, details):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            'INSERT INTO activity_log (user_id, action, details) VALUES (?, ?, ?)',
            (user_id, action, details)
        )
        conn.commit()
        conn.close()

    def save_sales_data(self, df):
        conn = sqlite3.connect(self.db_path)
        df.to_sql('sales_data', conn, if_exists='append', index=False)
        conn.close()

    def get_connection(self):
        return sqlite3.connect(self.db_path) 