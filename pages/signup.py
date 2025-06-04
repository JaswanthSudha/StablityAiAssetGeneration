import streamlit as st
from auth import create_user

def show():
    st.title("📝 Sign Up")
    
    with st.form("signup_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        password_confirm = st.text_input("Confirm Password", type="password")
        submit_button = st.form_submit_button("Sign Up")
        
        if submit_button:
            if username and email and password:
                if password != password_confirm:
                    st.error("Passwords don't match")
                else:
                    if "@" not in email or "." not in email:
                        st.error("Please enter a valid email address")
                    elif len(password) < 6:
                        st.error("Password should be at least 6 characters long")
                    else:
                        success, message = create_user(username, password, email)
                        if success:
                            st.success(message)
                            st.info("You can now log in with your credentials")
                        else:
                            st.error(message)
            else:
                st.error("Please fill in all fields")
    
    st.markdown("---")
    
    # Button for navigation to login
    if st.button("Already have an account? Login"):
        st.session_state['nav'] = "Login"
        st.rerun()