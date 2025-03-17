import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.auth.auth_handler import AuthHandler
from src.database.db_handler import DatabaseHandler

def create_admin():
    auth = AuthHandler()
    db = DatabaseHandler()
    
    # Create admin user if it doesn't exist
    if not auth.get_user('admin'):
        # Using a stronger password that meets requirements
        success = auth.register_user(
            username='admin',
            password='Admin@12345678!',  # Strong password meeting all requirements
            email='admin@shoplytics.com',
            role='admin'
        )
        if success:
            print("Admin user created successfully!")
        else:
            print("Failed to create admin user.")

if __name__ == "__main__":
    # Remove existing database
    if os.path.exists('data/shoplytics.db'):
        print("Removing existing database...")
        os.remove('data/shoplytics.db')
    
    create_admin() 