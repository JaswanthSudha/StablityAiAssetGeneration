import streamlit as st
from auth import login_user, set_authenticated

def show():
    st.title("🔐 Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")
        
        if submit_button:
            if username and password:
                success, result = login_user(username, password)
                if success:
                    set_authenticated(result, username)
                    st.success("Login successful!")
                    st.session_state['nav'] = "Project Dashboard"
                    st.rerun()
                else:
                    st.error(result)
            else:
                st.error("Please enter both username and password")
    
    st.markdown("---")
    
    # Button for navigation to signup
    if st.button("Don't have an account? Sign Up"):
        st.session_state['nav'] = "Sign Up"
        st.rerun()