import streamlit as st
from src.auth.auth_handler import AuthHandler
from src.database.db_handler import DatabaseHandler

def render_login():
    st.title("Login")
    
    db = DatabaseHandler()
    auth = AuthHandler()
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            user = db.get_user(username)
            if user:
                is_valid = auth.verify_password(password, user[2])
                
                if is_valid:
                    st.session_state.authenticated = True
                    st.session_state.user_id = user[0]
                    token = auth.encode_token(user[0])
                    st.session_state.token = token
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid password")
            else:
                st.error("User not found")

    if st.button("Create new account"):
        st.session_state.page = "register"
        st.rerun() 