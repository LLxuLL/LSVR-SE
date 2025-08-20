import os
from tqdm import tqdm
import trimesh
import numpy as np
import logging
from scipy.spatial import KDTree
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def conservative_triangulate(mesh):
    """Conservative triangulation: only subdivide polygon faces without changing geometry"""
    # Check if triangulation is needed
    if not mesh.is_empty and mesh.faces.shape[1] == 3:
        return mesh  # Already a triangular mesh

    # Only subdivide non-triangular faces
    if mesh.faces.shape[1] > 3:
        logging.info(f"Subdividing {mesh.faces.shape[0]} polygon faces")
        # Use more precise triangulation method
        return mesh.triangulate(engine='scipy')

    return mesh


def repair_mesh_with_detail_preservation(mesh):
    """
    Optimized mesh repair: repair method that preserves original details
    Use trimesh's built-in methods for safe repair
    """
    # Backup original vertices and faces
    orig_vertices = np.copy(mesh.vertices)
    orig_faces = np.copy(mesh.faces)

    try:
        # 1. Repair non-manifold geometry
        if not mesh.is_watertight or not mesh.is_winding_consistent:
            # Use safer repair methods
            mesh = fix_non_manifold(mesh)

        # 2. Repair holes (only small holes)
        if not mesh.is_watertight:
            # Use more reliable method to detect hole boundaries
            try:
                # Get boundary outline
                outline = mesh.outline()
                if outline:
                    # Calculate maximum boundary length
                    max_boundary_length = 0.0

                    # Try to get all boundary loops
                    if hasattr(outline, 'entities'):
                        # For newer versions of trimesh
                        for entity in outline.entities:
                            if hasattr(entity, 'length'):
                                # Check if vertex parameter is needed
                                try:
                                    if callable(entity.length) and entity.length.__code__.co_argcount > 1:
                                        length = entity.length(outline.vertices)
                                    elif callable(entity.length):
                                        length = entity.length()
                                    else:
                                        length = entity.length
                                except:
                                    # If unable to determine parameter count, try both ways
                                    try:
                                        length = entity.length(outline.vertices)
                                    except:
                                        try:
                                            length = entity.length()
                                        except:
                                            continue
                                max_boundary_length = max(max_boundary_length, length)
                    else:
                        # For older versions of trimesh
                        try:
                            if callable(outline.length) and outline.length.__code__.co_argcount > 1:
                                max_boundary_length = outline.length(outline.vertices)
                            elif callable(outline.length):
                                max_boundary_length = outline.length()
                            else:
                                max_boundary_length = outline.length
                        except:
                            # If unable to determine parameter count, try both ways
                            try:
                                max_boundary_length = outline.length(outline.vertices)
                            except:
                                try:
                                    max_boundary_length = outline.length()
                                except:
                                    pass

                    # Only repair small holes (less than 5% of model size)
                    if max_boundary_length > 0 and max_boundary_length < mesh.extents.max() * 0.05:
                        mesh.fill_holes()
            except Exception as e:
                logging.warning(f"Hole detection failed: {str(e)}, skipping hole repair")

        # 3. Remove isolated vertices (without changing shape)
        mesh.remove_unreferenced_vertices()

        # 4. Ensure mesh is manifold
        mesh.process(validate=True)

        # 5. Check repair result
        if mesh.is_empty:
            raise ValueError("Mesh is empty after repair")

        return mesh
    except Exception as e:
        logging.warning(f"Mesh repair failed: {str(e)}, using original mesh")
        return trimesh.Trimesh(vertices=orig_vertices, faces=orig_faces)


def fix_non_manifold(mesh):
    """Specialized method to fix non-manifold geometry (compatible with different trimesh versions)"""
    fixed_mesh = mesh.copy()

    # 1. Merge duplicate vertices
    fixed_mesh.merge_vertices()

    # 2. Repair non-manifold geometry - use more compatible method
    try:
        # Try using new repair methods
        if hasattr(trimesh.repair, 'fix_non_manifold'):
            fixed_mesh = trimesh.repair.fix_non_manifold(fixed_mesh)
        else:
            # Fallback to more basic methods
            trimesh.repair.fix_normals(fixed_mesh)
            trimesh.repair.fix_inversion(fixed_mesh)
            trimesh.repair.fix_winding(fixed_mesh)

            # Manually repair non-manifold edges
            fixed_mesh.process(validate=True)
    except Exception as e:
        logging.warning(f"Non-manifold repair failed: {str(e)}")

    # 3. Ensure triangulation
    if fixed_mesh.faces.shape[1] != 3:
        fixed_mesh = fixed_mesh.triangulate()

    return fixed_mesh


def align_repaired_mesh(repaired_mesh, orig_vertices, orig_faces):
    """
    Align repaired mesh with original mesh, preserving details
    Use ICP algorithm to align vertices, preserving original model details
    """
    # If vertex count is the same, return directly
    if len(repaired_mesh.vertices) == len(orig_vertices):
        return repaired_mesh

    try:
        # Create KDTree of original mesh for nearest point search
        tree = KDTree(orig_vertices)

        # For each repaired vertex, find the nearest original vertex
        _, indices = tree.query(repaired_mesh.vertices)

        # Calculate average displacement between original vertices and repaired vertices
        displacements = orig_vertices[indices] - repaired_mesh.vertices
        avg_displacement = np.mean(displacements, axis=0)

        # Apply average displacement correction
        aligned_vertices = repaired_mesh.vertices + avg_displacement

        # Create final mesh
        aligned_mesh = trimesh.Trimesh(
            vertices=aligned_vertices,
            faces=repaired_mesh.faces
        )

        # Calculate original bounding box
        orig_bbox = orig_vertices.max(axis=0) - orig_vertices.min(axis=0)

        # Calculate repaired bounding box
        repaired_bbox = aligned_vertices.max(axis=0) - aligned_vertices.min(axis=0)

        # Calculate scale factors and apply
        scale_factors = orig_bbox / (repaired_bbox + 1e-8)
        scale_matrix = np.diag([*scale_factors, 1.0])

        # Apply scaling
        aligned_mesh.apply_transform(scale_matrix)

        return aligned_mesh
    except Exception as e:
        logging.warning(f"Mesh alignment failed: {str(e)}, returning unaligned mesh")
        return repaired_mesh


def fix_obj_files(root_dir):
    """Repair OBJ files and save backups, preserving original shape details"""
    processed = 0
    skipped = 0
    fixed = 0

    # Ensure backup directory exists
    backup_dir = os.path.join(root_dir, "original_backups")
    os.makedirs(backup_dir, exist_ok=True)

    for class_name in tqdm(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir) or class_name == "original_backups":
            continue

        for fname in os.listdir(class_dir):
            if not fname.endswith('.obj'):
                continue

            obj_path = os.path.join(class_dir, fname)
            backup_path = os.path.join(backup_dir, f"original_{class_name}_{fname}")

            try:
                # Create original file backup (if it doesn't exist)
                if not os.path.exists(backup_path):
                    shutil.copy2(obj_path, backup_path)
                else:
                    # If backup already exists, make sure we use the original file
                    if os.path.exists(obj_path):
                        os.remove(obj_path)

                # Load from backup
                mesh = trimesh.load(backup_path)

                # Skip empty meshes
                if mesh.is_empty:
                    logging.warning(f"Skipping empty mesh: {backup_path}")
                    skipped += 1
                    continue

                # Record original vertex and face count
                orig_vertex_count = len(mesh.vertices)
                orig_face_count = len(mesh.faces)

                # Conservative triangulation (preserve details)
                mesh = conservative_triangulate(mesh)

                # Minimal repair (preserve details)
                mesh = repair_mesh_with_detail_preservation(mesh)

                # Check repaired mesh
                if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                    logging.error(f"Repair failed: {backup_path}")
                    skipped += 1
                    # Restore original file
                    shutil.copy2(backup_path, obj_path)
                    continue

                # Save repaired mesh
                mesh.export(obj_path)

                # Statistics
                processed += 1
                if orig_vertex_count != len(mesh.vertices) or orig_face_count != len(mesh.faces):
                    fixed += 1
                    logging.info(
                        f"Repaired {class_name}/{fname} [Vertices: {orig_vertex_count}→{len(mesh.vertices)}, Faces: {orig_face_count}→{len(mesh.faces)}]")
                else:
                    logging.info(f"No changes {class_name}/{fname} [Vertices: {orig_vertex_count}, Faces: {orig_face_count}]")

            except Exception as e:
                logging.error(f"Failed to process {obj_path}: {str(e)}")
                skipped += 1
                # Restore original file
                if os.path.exists(backup_path) and not os.path.exists(obj_path):
                    shutil.copy2(backup_path, obj_path)

    logging.info(f"Processing completed: Processed {processed} files, Repaired {fixed}, Skipped {skipped}")
    return processed, fixed, skipped


if __name__ == "__main__":
    dataset_path = "../data/shape_prior_dataset/full"
    logging.info(f"Starting dataset repair: {dataset_path}")
    fix_obj_files(dataset_path)