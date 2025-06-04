import torch
import argparse
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
import sys
import subprocess
import threading

# Load environment variables from .env file (for API key)
load_dotenv()

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
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32
    )
    
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
    api_key = "sk-PHKETwba1WqOf7CPhgPT0Gza6ePylGO3AQHrzq5u55NDrg2J"
    
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
            
            # Visualization is handled separately in the UI code
            if visualize and not st._is_running_with_streamlit:
                try:
                    # Import and use the display_mesh function directly
                    from displayMesh import display_mesh
                    display_mesh(output_path)
                except Exception as e:
                    print(f"Error visualizing 3D model: {str(e)}")
            
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

def main_streamlit():
    """Run the Streamlit UI version of the app"""
    
    # Set page config
    st.set_page_config(page_title="Stable Diffusion Generator", layout="wide")
    
    # Title and description
    st.title("🖼️ Stable Diffusion Image Generator")
    st.markdown("Generate images using Stable Diffusion and convert to 3D models")
    
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
        
        # Save the image to organized folder structure
        image_path, generation_folder = save_image(image, "output", model_id)
        
        # Save generation parameters for reproducibility
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
            "image_path": image_path
        }
        
        metadata_path = os.path.join(generation_folder, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
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
                    # Generate 3D model and save to the same folder as the image
                    output_path = os.path.join(generation_folder, f"model_{int(time.time())}.glb")
                    model_path = generate_3d_model(image_path, output_path=output_path, visualize=False)
                    
                    # Store path in session state
                    if 'current_model_path' not in st.session_state:
                        st.session_state['current_model_path'] = {}
                    session_key = f"model_{int(time.time())}"
                    st.session_state['current_model_path'][session_key] = model_path
                    
                    # Success message
                    model3d_area.success(f"3D model generated successfully!")
                    
                    # Create columns for the two buttons
                    col3d_1, col3d_2 = model3d_area.columns(2)
                    
                    # Button 1: Visualize the 3D model
                    view_key = f"view_{int(time.time())}"
                    if col3d_1.button("👁️ Visualize 3D Model", key=view_key):
                        try:
                            # Import the display_mesh function
                            from displayMesh import display_mesh
                            
                            # Use threading to avoid blocking the UI
                            def show_in_thread(path):
                                display_mesh(path)
                            
                            thread = threading.Thread(target=show_in_thread, args=(model_path,))
                            thread.daemon = True
                            thread.start()
                            
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
                            key=f"download_{int(time.time())}"
                        )
                    except Exception as e:
                        col3d_2.error(f"Error preparing download: {str(e)}")
                    
                    # Add path information
                    model3d_area.info(f"Files saved to: {generation_folder}")
                    
                except Exception as e:
                    model3d_area.error(f"Failed to generate 3D model: {str(e)}")
    elif not prompt and generate_btn:
        st.warning("Please enter a prompt to generate an image.")
    else:
        # Initial placeholder
        image_display.info("Enter a prompt and click 'Generate Image' to create your image.")

def main_cli():
    """Run the command line version of the app"""
    parser = argparse.ArgumentParser(description="Generate images with Stable Diffusion and convert to 3D models")
    
    # Basic arguments
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt (what to avoid)")
    parser.add_argument("--output_dir", type=str, default="generated_images", help="Directory to save generated images")
    parser.add_argument("--generate_3d", action="store_true", help="Generate 3D model from the image")
    
    # Model and generation parameters
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5", 
                        help="Model ID to use for generation")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (-1 for random)")
    
    # Post-processing options
    parser.add_argument("--upscale", type=float, default=1.0, help="Upscale factor")
    parser.add_argument("--sharpen", action="store_true", help="Apply sharpening")
    parser.add_argument("--brightness", type=float, default=1.0, help="Brightness adjustment")
    parser.add_argument("--contrast", type=float, default=1.0, help="Contrast adjustment")
    
    args = parser.parse_args()
    
    # Set up post-processing options
    post_process = {}
    if args.upscale != 1.0:
        post_process["upscale"] = args.upscale
    if args.sharpen:
        post_process["sharpen"] = True
    if args.brightness != 1.0:
        post_process["brightness"] = args.brightness
    if args.contrast != 1.0:
        post_process["contrast"] = args.contrast
    
    # Generate the image
    image, actual_seed, elapsed = generate_image(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        model_id=args.model,
        steps=args.steps,
        guidance_scale=args.guidance,
        width=args.width,
        height=args.height,
        seed=args.seed,
        post_process=post_process
    )
    
    # Save the image
    image_path, generation_folder = save_image(image, args.output_dir, args.model)
    
    # Save generation parameters for reproducibility
    metadata = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "model": args.model,
        "steps": args.steps,
        "guidance": args.guidance,
        "width": args.width,
        "height": args.height,
        "seed": actual_seed,
        "post_processing": post_process
    }
    
    metadata_path = os.path.splitext(image_path)[0] + "_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Generate 3D model if requested
    if args.generate_3d:
        try:
            # Don't auto-visualize to avoid crashing the CLI 
            model_path = generate_3d_model(image_path, visualize=False)
            print(f"Complete pipeline: Image -> 3D model")
            print(f"Image: {image_path}")
            print(f"3D Model: {model_path}")
            
            print("\nTo view the 3D model, you can:")
            print(f"1. Run: python displayMesh.py {model_path}")
            print("2. Open the GLB file with an external 3D viewer")
            print("3. Drag and drop the file into an online viewer like https://gltf-viewer.donmccurdy.com/")
        except Exception as e:
            print(f"Failed to generate 3D model: {e}")
    else:
        print(f"Image generation complete: {image_path}")
        print("To generate a 3D model, run with --generate_3d flag")

if __name__ == "__main__":
    # Check if script is run with command line arguments
    import sys
    
    if len(sys.argv) > 1:
        main_cli()
    else:
        # No command line arguments, run the Streamlit UI
        main_streamlit()