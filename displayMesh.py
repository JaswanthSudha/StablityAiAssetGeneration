import trimesh
import sys
import os
import numpy as np

def display_mesh(glb_path):
    """Display a 3D mesh from a GLB file using Trimesh"""
    try:
        if not os.path.exists(glb_path):
            print(f"File not found: {glb_path}")
            return False
        
        print(f"Loading 3D model: {glb_path}")
        
        # Try to load as a scene first (which is more common for GLB/GLTF files)
        try:
            # Load as scene which is more appropriate for GLB files
            scene = trimesh.load(glb_path, force='scene')
            
            # Check if scene has geometry
            if len(scene.geometry) > 0:
                print(f"Loaded scene with {len(scene.geometry)} geometry items")
                print(f"Displaying model: {glb_path}")
                scene.show()
                return True
            else:
                print("Scene contains no geometry.")
        except Exception as scene_error:
            print(f"Error loading as scene: {str(scene_error)}")
        
        # Fallback to loading as a single mesh
        try:
            # Try loading as a mesh
            mesh = trimesh.load(glb_path)
            
            if isinstance(mesh, trimesh.Trimesh):
                # It's a single mesh
                if len(mesh.faces) > 0:
                    print(f"Loaded mesh with {len(mesh.faces)} faces")
                    print(f"Displaying model: {glb_path}")
                    mesh.show()
                    return True
                else:
                    print("Mesh contains no faces.")
            elif isinstance(mesh, list) and len(mesh) > 0:
                # It's a list of meshes
                print(f"Loaded {len(mesh)} meshes")
                # Create a scene from the meshes
                scene = trimesh.Scene()
                for i, m in enumerate(mesh):
                    if isinstance(m, trimesh.Trimesh) and len(m.faces) > 0:
                        scene.add_geometry(m, node_name=f"mesh_{i}")
                
                if len(scene.geometry) > 0:
                    print(f"Displaying scene with {len(scene.geometry)} meshes")
                    scene.show()
                    return True
                else:
                    print("No valid meshes found in the list.")
            else:
                print(f"Loaded object is of type {type(mesh)}, not a standard mesh.")
        except Exception as mesh_error:
            print(f"Error loading as mesh: {str(mesh_error)}")
        
        # If we get here, nothing worked
        print("The model does not contain any valid mesh data that can be displayed.")
        return False
        
    except Exception as e:
        print(f"Error displaying mesh: {str(e)}")
        return False

# If the script is run directly, use command line arguments
if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        display_mesh(file_path)
    else:
        print("Please provide path to a .glb file as command line argument")
        print("Example: python displayMesh.py path/to/model.glb")
