import sys
import os
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.auth.auth_handler import AuthHandler
from src.database.db_handler import DatabaseHandler

def reset_database():
    # Remove existing database
    db_path = 'data/shoplytics.db'
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Initialize database
    db = DatabaseHandler()
    auth = AuthHandler()
    
    # Create admin user
    success = auth.register_user(
        username='admin',
        password='admin123',
        email='admin@shoplytics.com',
        role='admin'
    )
    
    if success:
        print("Database reset and admin user created successfully!")
        
        # Verify user was created
        user = auth.get_user('admin')
        if user:
            print("\nAdmin user details:")
            print(f"ID: {user[0]}")
            print(f"Username: {user[1]}")
            print(f"Email: {user[3]}")
            print(f"Role: {user[4]}")
        else:
            print("Warning: Admin user not found after creation!")
    else:
        print("Failed to create admin user!")

if __name__ == "__main__":
    reset_database() 