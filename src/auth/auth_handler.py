import jwt  # PyJWT package
import datetime
import streamlit as st
from passlib.hash import pbkdf2_sha256
from src.database.db_handler import DatabaseHandler

class AuthHandler:
    def __init__(self):
        self.db = DatabaseHandler()
        self.secret_key = st.secrets["jwt_secret"]  # Store this securely

    def hash_password(self, password):
        return pbkdf2_sha256.hash(password)

    def verify_password(self, password, hashed_password):
        return pbkdf2_sha256.verify(password, hashed_password)

    def encode_token(self, user_id):
        payload = {
            'exp': datetime.datetime.utcnow() + datetime.timedelta(days=1),
            'iat': datetime.datetime.utcnow(),
            'sub': user_id
        }
        # Convert payload values to strings
        string_payload = {k: str(v) for k, v in payload.items()}
        return jwt.encode(string_payload, self.secret_key, algorithm='HS256')

    def decode_token(self, token):
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload['sub']
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def get_user(self, username):
        return self.db.get_user(username)

    def register_user(self, username, password, email, role='user'):
        password_hash = self.hash_password(password)
        return self.db.add_user(username, password_hash, email, role) 