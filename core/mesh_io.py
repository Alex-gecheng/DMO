import trimesh
import numpy as np
import torch


def load_mesh(file_path):
    """
    Load a 3D mesh from the specified file path.

    Parameters:
    file_path (str): The path to the 3D mesh file.

    Returns:
    tuple: A tuple containing the vertices (V) and faces (F) of the mesh.
           V is a numpy array of shape (n_vertices, 3) and F is a numpy array of shape (n_faces, 3).
    """
    try:
        mesh = trimesh.load(file_path)
        print(f"Mesh loaded successfully from {file_path}")
        V=np.asarray(mesh.vertices,dtype=np.float32)
        F=np.asarray(mesh.faces,dtype=np.int32)
        return V,F
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return None
    

    
def save_mesh(file_path, vertices, faces):
    """
    Save a 3D mesh to the specified file path.

    Parameters:
    file_path (str): The path where the 3D mesh will be saved.
    vertices (numpy.ndarray): A numpy array of shape (n_vertices, 3) containing the vertex positions.
    faces (numpy.ndarray): A numpy array of shape (n_faces, 3) containing the indices of the vertices that form each face.

    Returns:
    bool: True if the mesh was saved successfully, False otherwise.
    """
    try:
        if torch.is_tensor(vertices):
            vertices = vertices.detach().cpu().numpy()
        if torch.is_tensor(faces):
            faces = faces.detach().cpu().numpy()
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(file_path)
        print(f"Mesh saved successfully to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving mesh: {e}")
        return False