from src.auth.auth_handler import AuthHandler
from src.database.db_handler import DatabaseHandler

def create_admin():
    auth = AuthHandler()
    db = DatabaseHandler()
    
    # Create admin user if it doesn't exist
    if not db.get_user('admin'):
        password_hash = auth.hash_password('admin123')
        db.add_user('admin', password_hash, 'admin@shoplytics.com', role='admin')

if __name__ == "__main__":
    create_admin() 