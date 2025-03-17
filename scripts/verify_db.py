import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.db_handler import DatabaseHandler

def verify_database():
    db = DatabaseHandler()
    
    # Check if database file exists
    print(f"\nChecking database file...")
    if os.path.exists(db.db_path):
        print(f"Database exists at: {db.db_path}")
        print(f"Database size: {os.path.getsize(db.db_path)} bytes")
    else:
        print("Database file not found!")
        return
    
    # Check tables
    conn = db.get_connection()
    cursor = conn.cursor()
    
    # Get list of tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("\nTables in database:")
    for table in tables:
        print(f"- {table[0]}")
    
    # Check users table
    print("\nUsers in database:")
    cursor.execute("SELECT * FROM users")
    users = cursor.fetchall()
    if users:
        for user in users:
            print(f"ID: {user[0]}")
            print(f"Username: {user[1]}")
            print(f"Email: {user[3]}")
            print(f"Role: {user[4]}")
            print("---")
    else:
        print("No users found in database!")
    
    conn.close()

if __name__ == "__main__":
    verify_database() 