import streamlit as st
from src.auth.auth_handler import AuthHandler
import re

def is_strong_password(password):
    """Check if password meets security requirements"""
    if len(password) < 12:  # Increased minimum length
        return False, "Password must be at least 12 characters long"
    
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"
    
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter"
    
    if not re.search(r"\d", password):
        return False, "Password must contain at least one number"
    
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Password must contain at least one special character"
    
    return True, "Password meets security requirements"

def render_register():
    st.title("Create Account")
    
    auth = AuthHandler()
    
    with st.form("register_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        
        # Add password requirements info
        st.markdown("""
        **Password Requirements:**
        - At least 12 characters long
        - Must contain uppercase and lowercase letters
        - Must contain numbers
        - Must contain special characters (!@#$%^&*(),.?":{}|<>)
        """)
        
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Register")
        
        if submitted:
            if not username or not email or not password:
                st.error("All fields are required!")
                return
                
            if password != confirm_password:
                st.error("Passwords do not match!")
                return
            
            # Check password strength
            is_strong, message = is_strong_password(password)
            if not is_strong:
                st.error(message)
                return
            
            # Validate email format
            if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                st.error("Please enter a valid email address!")
                return
            
            success = auth.register_user(username, password, email)
            
            if success:
                st.success("Account created successfully!")
                st.session_state.page = "login"
                st.rerun()
            else:
                st.error("Username or email already exists!")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Already have an account? Login"):
            st.session_state.page = "login"
            st.rerun() 