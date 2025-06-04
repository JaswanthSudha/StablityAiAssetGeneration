import streamlit as st
import sqlite3
import os
from auth import get_user_projects, create_project, delete_project

def show():
    st.title("🗂️ Your Projects")
    
    # Get user projects
    projects = get_user_projects(st.session_state['user_id'])
    
    # Create new project form
    with st.expander("Create New Project", expanded=False):
        with st.form("create_project_form"):
            project_name = st.text_input("Project Name")
            project_description = st.text_area("Description (optional)")
            submit_button = st.form_submit_button("Create Project")
            
            if submit_button and project_name:
                success, result = create_project(st.session_state['user_id'], project_name, project_description)
                if success:
                    st.session_state['current_project_id'] = result
                    st.session_state['current_project_name'] = project_name
                    st.success(f"Project '{project_name}' created successfully!")
                    st.rerun()
                else:
                    st.error(result)
    
    # Display projects as cards
    st.subheader("Your Projects")
    
    if not projects:
        st.info("You don't have any projects yet. Create one to get started!")
    else:
        # Create grid layout for projects
        cols = st.columns(3)
        
        for i, project in enumerate(projects):
            project_id, project_name, description, created_at = project
            with cols[i % 3]:
                with st.container():
                    st.markdown(f"### {project_name}")
                    st.markdown(f"*{description}*")
                    st.text(f"Created: {created_at}")
                    
                    # Get generations count
                    conn = sqlite3.connect('user_data.db')
                    c = conn.cursor()
                    c.execute("SELECT COUNT(*) FROM generations WHERE project_id = ?", (project_id,))
                    count = c.fetchone()[0]
                    conn.close()
                    
                    st.text(f"Generations: {count}")
                    
                    # Project actions - use columns for button layout
                    action_col1, action_col2 = st.columns(2)
                    
                    # Open project button
                    open_btn = action_col1.button(f"✅ Open", key=f"open_{project_id}")
                    if open_btn:
                        # Store the selected project data in session state
                        st.session_state['current_project_id'] = project_id
                        st.session_state['current_project_name'] = project_name
                        # Set the navigation to "Current Project"
                        st.session_state['nav'] = "Current Project"
                        # Force rerun
                        st.rerun()
                    
                    # Delete project button (with confirmation)
                    delete_btn = action_col2.button("🗑️ Delete", key=f"delete_{project_id}")
                    if delete_btn:
                        # Store project ID for confirmation
                        st.session_state['delete_project_id'] = project_id
                        st.session_state['delete_project_name'] = project_name
                        st.session_state['confirm_delete'] = True
                        st.rerun()
                
                st.markdown("---")
        
        # Handle project deletion confirmation
        if st.session_state.get('confirm_delete'):
            project_id = st.session_state['delete_project_id']
            project_name = st.session_state['delete_project_name']
            
            # Create a confirmation dialog
            st.warning(f"Are you sure you want to delete project '{project_name}'? This cannot be undone.")
            col1, col2 = st.columns(2)
            
            if col1.button("✅ Yes, Delete Project"):
                # Delete the project
                success = delete_project(project_id)
                if success:
                    st.success(f"Project '{project_name}' has been deleted.")
                    # Clear session state
                    st.session_state.pop('delete_project_id', None)
                    st.session_state.pop('delete_project_name', None)
                    st.session_state.pop('confirm_delete', None)
                    
                    # If current project was deleted, reset current project
                    if st.session_state.get('current_project_id') == project_id:
                        st.session_state['current_project_id'] = None
                        st.session_state['current_project_name'] = None
                    
                    st.rerun()
                else:
                    st.error("Failed to delete project. Please try again.")
            
            if col2.button("❌ Cancel"):
                # Clear session state
                st.session_state.pop('delete_project_id', None)
                st.session_state.pop('delete_project_name', None)
                st.session_state.pop('confirm_delete', None)
                st.rerun()