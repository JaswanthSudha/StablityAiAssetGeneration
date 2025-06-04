import torch
import time
import os
import streamlit as st
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageEnhance, ImageFilter
import requests
import json
from dotenv import load_dotenv
import io
import base64
import threading
import subprocess
import sys
import sqlite3
import hashlib
import uuid

# Import auth functions
from auth import (init_db, create_user, login_user, init_auth_state, is_authenticated, 
                 set_authenticated, logout, get_user_projects, create_project, 
                 save_generation, get_project, get_project_generations, delete_project)

# Load environment variables from .env file (for API key)
load_dotenv()

# Initialize database and auth state
init_db()
init_auth_state()

@st.cache_resource
def load_pipeline(model_id):
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    return pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def generate_image(
    prompt,
    negative_prompt="",
    model_id="runwayml/stable-diffusion-v1-5",
    steps=20,
    guidance_scale=7.5,
    width=512,
    height=512,
    seed=-1,
    post_process=None,
    progress_callback=None,
):
    """Generate an image using Stable Diffusion"""
    print(f"Loading model: {model_id}...")
    
    # Initialize the pipeline
    pipe=load_pipeline(model_id)
    
    # Use GPU if available, otherwise CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    print(f"Using device: {device}")
    
    # Set up generator with seed if provided
    generator = None
    if seed >= 0:
        generator = torch.Generator(device=device).manual_seed(seed)
        print(f"Using seed: {seed}")
    else:
        # Use a random seed and record it for reproducibility
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator(device=device).manual_seed(seed)
        print(f"Generated random seed: {seed}")
    
    # Start timing
    start_time = time.time()
    
    print(f"Generating image for prompt: '{prompt}'")
    
    # Define callback if needed
    callback_fn = None
    if progress_callback:
        def callback_fn(step, timestep, latents):
            progress_callback(step / steps)
            return
    
    # Generate the image
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator,
        callback=callback_fn,
        callback_steps=1
    ).images[0]
    
    # Calculate elapsed time
    elapsed = time.time() - start_time
    print(f"Image generated in {elapsed:.2f} seconds")
    
    # Post-processing if specified
    if post_process:
        if "upscale" in post_process:
            scale = post_process["upscale"]
            image = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)
        if "sharpen" in post_process and post_process["sharpen"]:
            image = image.filter(ImageFilter.SHARPEN)
        if "brightness" in post_process:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(post_process["brightness"])
        if "contrast" in post_process:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(post_process["contrast"])
    
    return image, seed, elapsed

def generate_3d_model(image_path, output_path=None, visualize=True):
    """Generate a 3D model from an image using the Stability AI API"""
    
    # Get API key from environment variable
    api_key = os.getenv("STABILITY_API_KEY") or "sk-kGzCmhtDfkylRJjUClfNGYVxhpipWEc69KeyJzfeTMMwiXC4"
    
    if not api_key:
        raise ValueError("API key not found. Please set the STABILITY_API_KEY environment variable.")
    
    print(f"Generating 3D model from image: {image_path}")
    
    # Default output path if not specified
    if not output_path:
        output_path = os.path.splitext(image_path)[0] + ".glb"
    
    try:
        response = requests.post(
            f"https://api.stability.ai/v2beta/3d/stable-fast-3d",
            headers={
                "authorization": f"Bearer {api_key}",
            },
            files={
                "image": open(image_path, "rb")
            },
            data={},
        )
        
        if response.status_code == 200:
            with open(output_path, 'wb') as file:
                file.write(response.content)
            print(f"3D model successfully generated and saved to: {output_path}")
            
            return output_path
        else:
            error_message = f"API Error: {response.status_code}"
            try:
                error_message = str(response.json())
            except:
                pass
            print(f"Error generating 3D model: {error_message}")
            raise Exception(error_message)
    except Exception as e:
        print(f"Failed to generate 3D model: {str(e)}")
        raise

def save_image(image, output_dir="output", model_id="default"):
    """Save the generated image to an organized output folder"""
    
    # Ensure base output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a unique folder for this generation
    timestamp = int(time.time())
    generation_folder = os.path.join(output_dir, f"generation_{timestamp}")
    if not os.path.exists(generation_folder):
        os.makedirs(generation_folder)
    
    # Save the image in the generation folder
    image_filename = f"image_{timestamp}.png"
    image_path = os.path.join(generation_folder, image_filename)
    
    # Save the image
    image.save(image_path)
    print(f"Image saved to: {image_path}")
    
    return image_path, generation_folder

def get_image_download_link(img, filename, text):
    """Generate a download link for an image"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">📥 {text}</a>'
    return href

# Login page
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
                    st.rerun()  # Changed from st.experimental_rerun()
                else:
                    st.error(result)
            else:
                st.error("Please enter both username and password")
    
    st.markdown("---")
    st.markdown("Don't have an account? [Create one](#signup)")

# Sign-up page
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

# Projects dashboard
def projects_dashboard():
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

# Project view showing generations
def project_view(project_id):
    # Get project details
    project = get_project(project_id)
    if not project:
        st.error("Project not found")
        return
    
    project_name = project[2]
    project_desc = project[3]
    
    # Get generations for this project
    generations = get_project_generations(project_id)
    
    # If there are no generations, redirect to Image Generator
    if not generations:
        # Store the original navigation to return later
        if 'previous_nav' not in st.session_state:
            st.session_state['previous_nav'] = st.session_state['nav']
            
        # Store empty project info for context
        st.session_state['empty_project_id'] = project_id
        st.session_state['empty_project_name'] = project_name
        
        # Navigate to Image Generator
        st.session_state['nav'] = "Image Generator"
        st.rerun()
        return
    
    # Display project header
    st.title(f"📁 Project: {project_name}")
    st.markdown(f"*{project_desc}*")
    
    # Button to return to project list
    if st.button("← Back to Projects"):
        st.session_state['current_project_id'] = None
        st.session_state['current_project_name'] = None
        st.session_state['nav'] = "Project Dashboard"  # Set nav back to dashboard
        st.rerun()
    
    # Display generations gallery
    st.subheader(f"Generations ({len(generations)})")
    
    # Create grid for generations
    cols = st.columns(3)
    
    for i, gen in enumerate(generations):
        gen_id, prompt, neg_prompt, model, steps, guidance, width, height, seed, img_path, model_path, created_at = gen
        
        # Check if image exists
        if os.path.exists(img_path):
            with cols[i % 3]:
                # Display image
                with Image.open(img_path) as img:
                    st.image(img, use_container_width=True)
                    
                    # Show prompt
                    st.markdown(f"**Prompt:** {prompt[:50]}..." if len(prompt) > 50 else f"**Prompt:** {prompt}")
                    
                    # Find metadata file
                    metadata_path = os.path.join(os.path.dirname(img_path), "metadata.json")
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            
                            # Create an expandable section to show metadata
                            with st.expander("📋 View Metadata"):
                                # Format the JSON for display
                                st.json(metadata)
                                
                                # Add a download button for the metadata JSON
                                with open(metadata_path, 'rb') as f:
                                    metadata_bytes = f.read()
                                
                                st.download_button(
                                    label="📥 Download Metadata",
                                    data=metadata_bytes,
                                    file_name=f"metadata_{gen_id}.json",
                                    mime="application/json",
                                    key=f"meta_{gen_id}"
                                )
                        except Exception as e:
                            st.error(f"Error loading metadata: {str(e)}")
                    
                    # Show 3D model badge if available
                    if model_path and os.path.exists(model_path):
                        st.success("3D Model Available")
                        
                        # Buttons for 3D model
                        col1, col2 = st.columns(2)
                        
                        # Visualize button
                        if col1.button("👁️ View 3D", key=f"view_{gen_id}"):
                            try:
                                script_dir = os.path.dirname(os.path.abspath(__file__))
                                display_script = os.path.join(script_dir, "displayMesh.py")
                                subprocess.Popen([sys.executable, display_script, model_path])
                                st.info("3D viewer launched")
                            except Exception as e:
                                st.error(f"Error: {e}")
                        
                        # Download button
                        with open(model_path, 'rb') as f:
                            model_bytes = f.read()
                        
                        download_name = os.path.basename(model_path)
                        col2.download_button(
                            label="📥 Download",
                            data=model_bytes,
                            file_name=download_name,
                            mime="model/gltf-binary",
                            key=f"dl_{gen_id}"
                        )
                
                # Add a divider between generations
                st.markdown("---")

# Image generator view
def generator_view():
    st.title("🖼️ Image Generator")
    
    # Show current project context
    if st.session_state.get('current_project_id') and st.session_state.get('current_project_name'):
        st.info(f"Creating in project: {st.session_state['current_project_name']}")
        
        # If redirected from empty project, show extra context
        if st.session_state.get('empty_project_id') == st.session_state.get('current_project_id'):
            st.warning("This project has no generations yet. Create your first image!")
            
            # Add a button to go back to previous navigation if available
            if st.session_state.get('previous_nav'):
                if st.button("← Cancel and go back"):
                    st.session_state['nav'] = st.session_state['previous_nav']
                    st.session_state.pop('previous_nav', None)
                    st.session_state.pop('empty_project_id', None)
                    st.rerun()
    else:
        st.error("Please select or create a project first")
        return
    
    # Create a container for the UI
    with st.container():
        # Create two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Prompt input
            st.subheader("Image Settings")
            prompt = st.text_area("Enter your prompt", 
                                placeholder="A professional portrait of a middle-aged male architect, wearing a black turtleneck",
                                height=100)
            
            # Negative prompt
            negative_prompt = st.text_area("Negative prompt (what to avoid)", 
                                        placeholder="blurry, distorted, low quality",
                                        height=68)
            
            # Model selector
            model_id = st.selectbox("Model", [
                "CompVis/stable-diffusion-v1-4", 
                "runwayml/stable-diffusion-v1-5",
                "dreamlike-art/dreamlike-photoreal-2.0"
            ], index=1)
            
            # Advanced settings in an expander
            with st.expander("Advanced Settings", expanded=True):
                col_a, col_b = st.columns(2)
                with col_a:
                    num_steps = st.slider("Inference Steps", min_value=10, max_value=50, value=20, 
                                       help="Higher values = more detail, but slower")
                    guidance_scale = st.slider("Guidance Scale", min_value=1.0, max_value=15.0, value=7.5, step=0.5,
                                            help="How closely to follow the prompt")
                with col_b:
                    width = st.slider("Width", min_value=256, max_value=768, value=512, step=64)
                    height = st.slider("Height", min_value=256, max_value=768, value=512, step=64)
                    seed = st.number_input("Random Seed", value=-1, 
                                        help="Set to -1 for random, or use a specific value for reproducible results")
            
            # Post-processing options
            with st.expander("Post-Processing Options"):
                apply_upscale = st.checkbox("Upscale Image (2x)", value=False)
                apply_sharpen = st.checkbox("Sharpen Image", value=False)
                apply_brightness = st.slider("Adjust Brightness", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
                apply_contrast = st.slider("Adjust Contrast", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
            
            # 3D model generation options
            with st.expander("3D Model Generation"):
                enable_3d = st.checkbox("Generate 3D Model", value=False, 
                                      help="Uses Stability AI API to generate a 3D model from the image")
                st.info("Requires a Stability AI API key set in the .env file as STABILITY_API_KEY")
            
            # Generate button
            generate_btn = st.button("🚀 Generate Image", use_container_width=True)
        
        # Display area
        with col2:
            st.subheader("Generated Image")
            image_display = st.empty()
            info_display = st.empty()
            download_area = st.empty()
            model3d_area = st.empty()
            metadata_area = st.empty()
    
    # When generate button is clicked
    if generate_btn and prompt:
        # Set up post-processing options
        post_process = {}
        if apply_upscale:
            post_process["upscale"] = 2.0
        if apply_sharpen:
            post_process["sharpen"] = True
        if apply_brightness != 1.0:
            post_process["brightness"] = apply_brightness
        if apply_contrast != 1.0:
            post_process["contrast"] = apply_contrast
        
        # Show a spinner while loading the model
        with st.spinner("Loading model... (this may take a moment the first time)"):
            # Set up progress bar
            progress_bar = col2.progress(0)
            
            # Define progress callback function
            def update_progress(progress):
                progress_bar.progress(progress)
            
            # Generate the image
            image, actual_seed, elapsed = generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                model_id=model_id,
                steps=num_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                seed=seed,
                post_process=post_process,
                progress_callback=update_progress
            )
        
        # Save generation data
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "model": model_id,
            "steps": num_steps,
            "guidance": guidance_scale,
            "width": width,
            "height": height,
            "seed": actual_seed,
            "post_processing": post_process,
            "timestamp": int(time.time()),
        }
        
        # Save to the project
        try:
            image_path, generation_folder = save_generation(
                st.session_state['user_id'],
                st.session_state['current_project_id'],
                image,
                metadata
            )
            
            # Update metadata with paths
            metadata["image_path"] = image_path
            metadata["generation_folder"] = generation_folder
            
            # Get the latest generation ID for this image (for 3D model update)
            conn = sqlite3.connect('user_data.db')
            c = conn.cursor()
            c.execute("""
                SELECT id FROM generations 
                WHERE image_path = ? 
                ORDER BY created_at DESC LIMIT 1
            """, (image_path,))
            generation_id = c.fetchone()[0]
            conn.close()
            
            # Display the image and info
            image_display.image(image, use_container_width=True)
            info_display.success(f"Image generated in {elapsed:.1f} seconds using {num_steps} steps")
            
            # Display download links
            download_area.markdown(get_image_download_link(image, f"stable_diffusion_{int(time.time())}.png", "Download Image"), unsafe_allow_html=True)
            
            # Display prompt info
            metadata_area.code(f"""
Prompt: {prompt}
Negative prompt: {negative_prompt}
Model: {model_id}
Size: {width}x{height}
Steps: {num_steps}
Guidance Scale: {guidance_scale}
Seed: {actual_seed}
Output folder: {generation_folder}
            """)
            
            # Generate 3D model if requested
            if enable_3d:
                with st.spinner("Generating 3D model using Stability AI API..."):
                    try:
                        # Generate 3D model and save to the same folder
                        output_path = os.path.join(generation_folder, f"model_{int(time.time())}.glb")
                        model_path = generate_3d_model(image_path, output_path=output_path, visualize=False)
                        
                        # Update the generation record with model path
                        conn = sqlite3.connect('user_data.db')
                        c = conn.cursor()
                        c.execute("UPDATE generations SET model_path = ? WHERE id = ?", (model_path, generation_id))
                        conn.commit()
                        conn.close()
                        
                        # Store path in session state
                        st.session_state['current_model_path'] = model_path
                        st.session_state['model_generated'] = True
                        st.session_state['generation_folder'] = generation_folder
                        
                        # Success message
                        model3d_area.success(f"3D model generated successfully!")
                        
                        # Display buttons
                        col3d_1, col3d_2 = model3d_area.columns(2)
                        
                        # Button 1: Visualize the 3D model
                        if col3d_1.button("👁️ Visualize 3D Model", key=f"vis_{int(time.time())}"):
                            try:
                                script_dir = os.path.dirname(os.path.abspath(__file__))
                                display_script = os.path.join(script_dir, "displayMesh.py")
                                subprocess.Popen([sys.executable, display_script, model_path])
                                model3d_area.info("3D model viewer launched in a separate window")
                            except Exception as e:
                                model3d_area.error(f"Error displaying 3D model: {str(e)}")
                        
                        # Button 2: Download the 3D model
                        try:
                            with open(model_path, 'rb') as f:
                                model_bytes = f.read()
                            
                            download_name = os.path.basename(model_path)
                            col3d_2.download_button(
                                label="📥 Download 3D Model",
                                data=model_bytes,
                                file_name=download_name,
                                mime="model/gltf-binary",
                                key=f"dl_{int(time.time())}"
                            )
                        except Exception as e:
                            col3d_2.error(f"Error preparing download: {str(e)}")
                        
                        # Add path information
                        model3d_area.info(f"Files saved to: {generation_folder}")
                        
                    except Exception as e:
                        model3d_area.error(f"Failed to generate 3D model: {str(e)}")
                        st.session_state['model_generated'] = False
                        
        except Exception as e:
            st.error(f"Error saving generation: {str(e)}")
            
    elif not prompt and generate_btn:
        st.warning("Please enter a prompt to generate an image.")
    else:
        # Initial placeholder
        image_display.info("Enter a prompt and click 'Generate Image' to create your image.")

# Main app function
def main_streamlit():
    st.set_page_config(page_title="Stable Diffusion Project Manager", layout="wide")
    
    # Initialize nav in session state if not present 
    # or reset it to a valid value if it's not valid for the current state
    if not is_authenticated():
        if 'nav' not in st.session_state or st.session_state['nav'] not in ["Login", "Sign Up"]:
            st.session_state['nav'] = "Login"
    else:
        # If user is authenticated but nav is still set to Login or Sign Up, reset it
        if 'nav' not in st.session_state or st.session_state['nav'] in ["Login", "Sign Up"]:
            st.session_state['nav'] = "Project Dashboard"
    
    # Sidebar for navigation and user info
    with st.sidebar:
        st.title("Navigation")
        
        if is_authenticated():
            st.success(f"Logged in as: {st.session_state['username']}")
            
            # Navigation options
            if st.session_state.get('current_project_id'):
                available_options = ["Project Dashboard", "Current Project", "Image Generator"]
                # Ensure the current nav is in the available options
                if st.session_state['nav'] not in available_options:
                    st.session_state['nav'] = "Project Dashboard"
                    
                nav = st.radio("Go to", available_options,
                              index=available_options.index(st.session_state['nav']))
            else:
                available_options = ["Project Dashboard", "Image Generator"]
                # Ensure the current nav is in the available options
                if st.session_state['nav'] not in available_options:
                    st.session_state['nav'] = "Project Dashboard"
                    
                nav = st.radio("Go to", available_options,
                              index=available_options.index(st.session_state['nav']))
            
            # Update session state when nav changes from the radio buttons
            if nav != st.session_state['nav']:
                st.session_state['nav'] = nav
                st.rerun()  # Changed from st.experimental_rerun()

            # Logout button
            if st.button("Logout"):
                logout()
                st.session_state['nav'] = "Login"
                st.rerun()  # Changed from st.experimental_rerun()

        else:
            nav = st.radio("Account", ["Login", "Sign Up"],
                          index=0 if st.session_state['nav'] == "Login" else 1)
            if nav != st.session_state['nav']:
                st.session_state['nav'] = nav
                st.rerun()  # Changed from st.experimental_rerun()
    
    # Main content based on navigation
    if not is_authenticated():
        if st.session_state['nav'] == "Login":
            login_page()
        else:  # Sign Up
            signup_page()
    else:
        if st.session_state['nav'] == "Project Dashboard":
            projects_dashboard()
        elif st.session_state['nav'] == "Current Project" and st.session_state.get('current_project_id'):
            project_view(st.session_state['current_project_id'])
        else:  # Image Generator
            generator_view()

if __name__ == "__main__":
    main_streamlit()