import open3d as o3d
import sys
import os

def display_mesh(glb_path):
    """Display a 3D mesh from a GLB file"""
    try:
        if not os.path.exists(glb_path):
            print(f"File not found: {glb_path}")
            return False
        
        print(f"Loading 3D model: {glb_path}")
        mesh = o3d.io.read_triangle_mesh(glb_path)
        
        if not mesh.has_triangles():
            print("The model does not contain any triangle mesh.")
            return False
        
        print(f"Displaying model: {glb_path}")
        o3d.visualization.draw_geometries([mesh])
        return True
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
