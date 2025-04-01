import numpy as np
from biopandas.pdb import PandasPdb
import os

def align_protein(pdb_file, output_file):
    # Load PDB file into Pandas DataFrame
    ppdb = PandasPdb().read_pdb(pdb_file)
    # get only the first model in the pdb file
    try:
        ppdb = ppdb.get_model(1)
    except:
        pass
    atom_df = ppdb.df['ATOM']
    
    # Extract atomic coordinates
    coords = atom_df[['x_coord', 'y_coord', 'z_coord']].to_numpy()
    
    # Step 1: Align longest dimension with z-axis
    centroid = np.mean(coords, axis=0)
    coords_centered = coords - centroid  # Center coordinates at origin
    cov_matrix = np.cov(coords_centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Longest dimension corresponds to eigenvector with largest eigenvalue
    longest_axis = eigenvectors[:, np.argmax(eigenvalues)]
    target_axis_z = np.array([0, 0, 1])  # Target is z-axis
    
    # Compute rotation matrix to align longest axis with z-axis
    rotation_matrix_1 = compute_rotation_matrix(longest_axis, target_axis_z)
    coords_rotated_1 = np.dot(coords_centered, rotation_matrix_1.T)
    
    # Step 2: Align interface with xz-plane
    interface_coords = extract_interface_coords(atom_df)  # Extract interface atoms
    interface_coords_centered = interface_coords - np.mean(interface_coords, axis=0)
    
    # Perform PCA on interface residues
    cov_matrix_interface = np.cov(interface_coords_centered.T)
    _, eigenvectors_interface = np.linalg.eig(cov_matrix_interface)
    
    # Second principal axis of interface should align with xz-plane (normal is y-axis)
    second_axis_interface = eigenvectors_interface[:, 1]
    target_axis_y = np.array([0, 1, 0])  # Normal to xz-plane
    
    # Compute rotation matrix to align second axis of interface normal to y-axis
    rotation_matrix_2 = compute_rotation_matrix(second_axis_interface, target_axis_y)
    
    # Combine both rotations and apply them sequentially
    combined_rotation_matrix = np.dot(rotation_matrix_2, rotation_matrix_1)
    final_coords = np.dot(coords_centered, combined_rotation_matrix.T) + centroid
    
    # Update DataFrame with transformed coordinates
    atom_df[['x_coord', 'y_coord', 'z_coord']] = final_coords
    
    # Save transformed structure to a new PDB file
    ppdb.to_pdb(path=output_file, records=['ATOM'], gz=False)

def compute_rotation_matrix(v1, v2):
    """
    Compute a rotation matrix that aligns vector v1 with vector v2.
    """
    v1 = v1 / np.linalg.norm(v1)  # Normalize vector v1
    v2 = v2 / np.linalg.norm(v2)  # Normalize vector v2
    
    cross_product = np.cross(v1, v2)
    dot_product = np.dot(v1, v2)
    
    if np.isclose(dot_product, 1.0):  # Vectors are already aligned
        return np.eye(3)
    
    if np.isclose(dot_product, -1.0):  # Vectors are opposite; rotate by 180 degrees
        orthogonal_vector = np.array([1, 0, 0]) if not np.allclose(v1, [1, 0, 0]) else np.array([0, 1, 0])
        cross_product = np.cross(v1, orthogonal_vector)
        cross_product /= np.linalg.norm(cross_product)
        skew_matrix = create_skew_symmetric(cross_product)
        return np.eye(3) + skew_matrix + skew_matrix @ skew_matrix
    
    skew_matrix = create_skew_symmetric(cross_product)
    
    rotation_matrix = (
        np.eye(3) +
        skew_matrix +
        skew_matrix @ skew_matrix * (1 - dot_product) / (np.linalg.norm(cross_product)**2)
    )
    
    return rotation_matrix

def create_skew_symmetric(vector):
    """
    Create a skew-symmetric matrix from a vector.
    """
    return np.array([
        [0, -vector[2], vector[1]],
        [vector[2], 0, -vector[0]],
        [-vector[1], vector[0], 0]
    ])

def extract_interface_coords(atom_df):
    """
    Extract coordinates of interface residues.
    
    This is a placeholder function. Replace it with your logic for identifying 
    interface residues based on proximity between chains or other criteria.
    
    Returns:
        numpy.ndarray: Coordinates of interface residues.
    """
    # Example: Select atoms from chain A within a certain distance of chain B
    chain_a_atoms = atom_df[atom_df['chain_id'] == 'A']
    chain_b_atoms = atom_df[atom_df['chain_id'] == 'B']
    
    chain_a_coords = chain_a_atoms[['x_coord', 'y_coord', 'z_coord']].to_numpy()
    chain_b_coords = chain_b_atoms[['x_coord', 'y_coord', 'z_coord']].to_numpy()
    
    distances = np.linalg.norm(chain_a_coords[:, None] - chain_b_coords[None, :], axis=-1)
    
    threshold_distance = 5.0  # Define distance threshold for interaction (e.g., 5 Å)
    
    interface_indices_a = np.any(distances < threshold_distance, axis=1)
    
    return chain_a_coords[interface_indices_a]

# # Example usage:
# align_protein_with_biopandas("input.pdb", "output.pdb")


if __name__ == "__main__":  
    pdb_files_dir = '../test_pdb_files'
    output_dir = './pdb_files'

    os.makedirs(output_dir, exist_ok=True)

    pdb_files = os.listdir(pdb_files_dir)
    for pdb_file in pdb_files:
        if pdb_file.endswith('.pdb'):
            input_path = os.path.join(pdb_files_dir, pdb_file)
            output_path = os.path.join(output_dir, f"{pdb_file}")
            align_protein(input_path, output_path)
            print(f"Aligned {pdb_file} and saved to {output_path}")
