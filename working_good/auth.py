import sqlite3
import hashlib
import uuid
import os
import streamlit as st
import time
import json
import shutil

# Initialize database
def init_db():
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    
    # Create users table if it doesn't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS users
    (id TEXT PRIMARY KEY, 
     username TEXT UNIQUE, 
     password TEXT,
     email TEXT UNIQUE,
     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
    ''')
    
    # Create projects table if it doesn't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS projects
    (id TEXT PRIMARY KEY,
     user_id TEXT,
     name TEXT,
     description TEXT,
     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
     last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
     FOREIGN KEY (user_id) REFERENCES users(id))
    ''')
    
    # Create generations table to store metadata about each generation
    c.execute('''
    CREATE TABLE IF NOT EXISTS generations
    (id TEXT PRIMARY KEY,
     project_id TEXT,
     user_id TEXT,
     prompt TEXT,
     negative_prompt TEXT,
     model TEXT,
     steps INTEGER,
     guidance REAL,
     width INTEGER,
     height INTEGER,
     seed INTEGER,
     image_path TEXT,
     model_path TEXT,
     metadata_path TEXT,
     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
     FOREIGN KEY (project_id) REFERENCES projects(id),
     FOREIGN KEY (user_id) REFERENCES users(id))
    ''')
    
    conn.commit()
    conn.close()

# Create a new user
def create_user(username, password, email):
    # Connect to DB
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    
    # Check if username already exists
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    if c.fetchone() is not None:
        conn.close()
        return False, "Username already exists"
    
    # Check if email already exists
    c.execute("SELECT * FROM users WHERE email = ?", (email,))
    if c.fetchone() is not None:
        conn.close()
        return False, "Email already exists"
    
    # Hash password
    salt = uuid.uuid4().hex
    hashed_pw = hashlib.sha256(salt.encode() + password.encode()).hexdigest()
    password_db = f"{salt}:{hashed_pw}"
    
    # Create UUID for user
    user_id = str(uuid.uuid4())
    
    try:
        c.execute("INSERT INTO users (id, username, password, email) VALUES (?, ?, ?, ?)", 
                 (user_id, username, password_db, email))
        
        # Create a default project for the user
        default_project_id = str(uuid.uuid4())
        c.execute("INSERT INTO projects (id, user_id, name, description) VALUES (?, ?, ?, ?)",
                 (default_project_id, user_id, "Default Project", "Your first project"))
        
        conn.commit()
        conn.close()
        
        # Create user directory structure
        os.makedirs(f"output/users/{user_id}", exist_ok=True)
        os.makedirs(f"output/users/{user_id}/{default_project_id}", exist_ok=True)
        
        return True, "User created successfully"
    except Exception as e:
        conn.close()
        return False, f"Error creating user: {str(e)}"

# Verify login
def login_user(username, password):
    # Connect to DB
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    
    # Get user from DB
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    
    if user is None:
        return False, "Invalid username or password"
    
    # Verify password
    salt, stored_pw = user[2].split(":")
    hashed_pw = hashlib.sha256(salt.encode() + password.encode()).hexdigest()
    
    if hashed_pw == stored_pw:
        return True, user[0]  # Return user_id
    else:
        return False, "Invalid username or password"

# Initialize authentication state
def init_auth_state():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'user_id' not in st.session_state:
        st.session_state['user_id'] = None
    if 'username' not in st.session_state:
        st.session_state['username'] = None
    if 'current_project_id' not in st.session_state:
        st.session_state['current_project_id'] = None
    if 'current_project_name' not in st.session_state:
        st.session_state['current_project_name'] = None

# Check authentication
def is_authenticated():
    return st.session_state['logged_in']

# Set authentication
def set_authenticated(user_id, username):
    st.session_state['logged_in'] = True
    st.session_state['user_id'] = user_id
    st.session_state['username'] = username
    
    # Set default project
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    c.execute("SELECT id, name FROM projects WHERE user_id = ? ORDER BY created_at DESC LIMIT 1", (user_id,))
    project = c.fetchone()
    conn.close()
    
    if project:
        st.session_state['current_project_id'] = project[0]
        st.session_state['current_project_name'] = project[1]

# Clear authentication
def logout():
    st.session_state['logged_in'] = False
    st.session_state['user_id'] = None
    st.session_state['username'] = None
    st.session_state['current_project_id'] = None
    st.session_state['current_project_name'] = None

# Get user projects
def get_user_projects(user_id):
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    c.execute("SELECT id, name, description, created_at FROM projects WHERE user_id = ? ORDER BY last_updated DESC", (user_id,))
    projects = c.fetchall()
    conn.close()
    
    return projects

# Create a new project
def create_project(user_id, name, description=""):
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    
    # Check if project name already exists for this user
    c.execute("SELECT * FROM projects WHERE user_id = ? AND name = ?", (user_id, name))
    if c.fetchone() is not None:
        conn.close()
        return False, "Project name already exists"
    
    project_id = str(uuid.uuid4())
    
    try:
        c.execute("INSERT INTO projects (id, user_id, name, description) VALUES (?, ?, ?, ?)",
                 (project_id, user_id, name, description))
        conn.commit()
        conn.close()
        
        # Create project directory
        os.makedirs(f"output/users/{user_id}/{project_id}", exist_ok=True)
        
        return True, project_id
    except Exception as e:
        conn.close()
        return False, f"Error creating project: {str(e)}"

# Get project details
def get_project(project_id):
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    c.execute("SELECT id, user_id, name, description, created_at FROM projects WHERE id = ?", (project_id,))
    project = c.fetchone()
    conn.close()
    return project

# Save generation to database and file system
def save_generation(user_id, project_id, image, metadata):
    generation_id = str(uuid.uuid4())
    timestamp = int(time.time())
    
    # Create project folder paths
    user_folder = f"output/users/{user_id}"
    project_folder = f"{user_folder}/{project_id}"
    generation_folder = f"{project_folder}/generation_{timestamp}"
    
    # Ensure directories exist
    os.makedirs(generation_folder, exist_ok=True)
    
    # Save image
    image_filename = f"image_{timestamp}.png"
    image_path = os.path.join(generation_folder, image_filename)
    image.save(image_path)
    
    # Save metadata
    metadata_path = os.path.join(generation_folder, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Add to database
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    
    try:
        c.execute('''
        INSERT INTO generations 
        (id, project_id, user_id, prompt, negative_prompt, model, steps, guidance, 
         width, height, seed, image_path, metadata_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            generation_id, 
            project_id, 
            user_id,
            metadata.get('prompt', ''),
            metadata.get('negative_prompt', ''),
            metadata.get('model', ''),
            metadata.get('steps', 0),
            metadata.get('guidance', 0.0),
            metadata.get('width', 0),
            metadata.get('height', 0),
            metadata.get('seed', 0),
            image_path,
            metadata_path
        ))
        
        # Update project last_updated time
        c.execute("UPDATE projects SET last_updated = CURRENT_TIMESTAMP WHERE id = ?", (project_id,))
        conn.commit()
        conn.close()
        
        return image_path, generation_folder
    except Exception as e:
        conn.close()
        raise e

# Update generation with 3D model
def update_generation_with_model(generation_id, model_path):
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    c.execute("UPDATE generations SET model_path = ? WHERE id = ?", (model_path, generation_id))
    conn.commit()
    conn.close()

# Get user generations by project
def get_project_generations(project_id):
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    c.execute('''
    SELECT id, prompt, negative_prompt, model, steps, guidance, width, height, 
           seed, image_path, model_path, created_at
    FROM generations 
    WHERE project_id = ? 
    ORDER BY created_at DESC
    ''', (project_id,))
    generations = c.fetchall()
    conn.close()
    return generations

import sqlite3
import hashlib
import uuid
import os
import streamlit as st
from auth import init_db, create_user, login_user, init_auth_state, is_authenticated, set_authenticated, logout

# Initialize database and authentication state
init_db()
init_auth_state()

def login_page():
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
                    st.session_state['nav'] = "Project Dashboard"  # Explicitly set nav before rerun
                    st.rerun()
                else:
                    st.error(result)
            else:
                st.error("Please enter both username and password")
    
    st.markdown("---")
    st.markdown("Don't have an account? [Create one](#signup)")

def signup_page():
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
    st.markdown("Already have an account? [Login](#login)")

def main_streamlit():
    """Run the Streamlit UI version of the app"""
    
    # Set page config
    st.set_page_config(page_title="Stable Diffusion Generator", layout="wide")
    
    # Add sidebar with user info and navigation
    with st.sidebar:
        st.title("Navigation")
        
        if is_authenticated():
            st.success(f"Logged in as: {st.session_state['username']}")
            if st.button("Logout"):
                logout()
                st.rerun()
            
            page = st.radio("Go to", ["Image Generator"])
        else:
            page = st.radio("Go to", ["Login", "Sign Up"])
    
    # Show appropriate page based on authentication and selection
    if not is_authenticated():
        if page == "Login":
            login_page()
        else:  # Sign Up
            signup_page()
    else:
        # Main application
        # Title and description
        st.title("🖼️ Stable Diffusion Image Generator")
        st.markdown("Generate images using Stable Diffusion and convert to 3D models")
        
        # Create a container for the UI
        with st.container():
            # Create two columns
            col1, col2 = st.columns([1, 1])
            
            # Continue with your existing UI
            with col1:
                # Prompt input
                st.subheader("Image Settings")
                prompt = st.text_area("Enter your prompt", 
                                    placeholder="A professional portrait of a middle-aged male architect, wearing a black turtleneck",
                                    height=100)
                
                # Rest of your existing UI code...
                # [Keep everything as is]

if __name__ == "__main__":
    main_streamlit()