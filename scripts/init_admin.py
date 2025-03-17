import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.auth.auth_handler import AuthHandler

def create_admin():
    auth = AuthHandler()
    
    # Create admin user if it doesn't exist
    if not auth.get_user('admin'):
        success = auth.register_user(
            username='admin',
            password='admin123',
            email='admin@shoplytics.com',
            role='admin'
        )
        if success:
            print("Admin user created successfully!")
        else:
            print("Failed to create admin user.")

if __name__ == "__main__":
    create_admin() 