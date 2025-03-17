import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.db_handler import DatabaseHandler

def check_database():
    db = DatabaseHandler()
    conn = db.get_connection()
    cursor = conn.cursor()
    
    # Check users table
    cursor.execute("SELECT * FROM users")
    users = cursor.fetchall()
    print("\nUsers in database:")
    for user in users:
        print(f"ID: {user[0]}, Username: {user[1]}, Email: {user[3]}, Role: {user[4]}")
    
    conn.close()

if __name__ == "__main__":
    check_database() 