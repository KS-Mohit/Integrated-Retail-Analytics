import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.db_handler import DatabaseHandler

def setup_database():
    # Remove existing database if it exists
    db_path = os.path.join(os.getcwd(), 'data', 'shoplytics.db')
    if os.path.exists(db_path):
        print(f"Removing existing database at {db_path}")
        os.remove(db_path)

    # Create new database
    db = DatabaseHandler()
    
    # Verify admin user
    conn = db.get_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ?', ('admin',))
    admin = c.fetchone()
    conn.close()

    if admin:
        print("\nAdmin user verified:")
        print(f"ID: {admin[0]}")
        print(f"Username: {admin[1]}")
        print(f"Email: {admin[3]}")
        print(f"Role: {admin[4]}")
    else:
        print("Warning: Admin user not found!")

if __name__ == "__main__":
    setup_database() 