# 文件: mesh_generation.py

import open3d as o3d
import numpy as np
import trimesh
from object_prior import ObjectPrior
import copy


def create_mesh(pcd, class_name="unknown", fit_strength=0.9):
    """Generate mesh from point cloud - point-to-point fitting version."""
    # Ensure the point cloud has normals
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

    # Copy the point cloud to avoid modifying the original data
    working_pcd = copy.deepcopy(pcd)

    # Apply shape prior to the point cloud - only once before mesh generation
    if fit_strength > 0:
        object_prior = ObjectPrior()
        # Pass the class_name parameter
        working_pcd = object_prior.apply_shape_prior(working_pcd, class_name, fit_strength)

    # Poisson reconstruction - adjust parameters
    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            working_pcd, depth=12, linear_fit=True, n_threads=4)
    except:
        # Fallback to Ball-Pivoting algorithm
        print("Poisson reconstruction failed, using Ball-Pivoting")
        radii = [0.005, 0.01, 0.02, 0.04]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            working_pcd, o3d.utility.DoubleVector(radii))

    # Remove low-density vertices (for Poisson reconstruction only)
    if densities is not None and len(densities) > 0:
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        # Keep more vertices
        mesh.remove_vertices_by_mask(vertices_to_remove)

    # Mesh simplification (to keep more triangles)
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=50000)

    # Mesh smoothing
    mesh = mesh.filter_smooth_taubin(number_of_iterations=15)

    # Compute vertex normals
    mesh.compute_vertex_normals()

    # Fix the mesh
    mesh = fix_mesh_topology(mesh)
    return mesh


def fix_mesh_topology(mesh):
    """Fix mesh topology issues."""
    # Convert to trimesh for fixing
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    # Create a trimesh object
    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Fix potential topology issues
    tri_mesh.process()

    # Convert back to Open3D format
    fixed_mesh = o3d.geometry.TriangleMesh()
    fixed_mesh.vertices = o3d.utility.Vector3dVector(tri_mesh.vertices)
    fixed_mesh.triangles = o3d.utility.Vector3iVector(tri_mesh.faces)
    fixed_mesh.compute_vertex_normals()
    return fixed_mesh


def apply_shape_prior(mesh, class_name, fit_strength=0.9):
    """Apply object shape prior to optimize the mesh - added fit_strength parameter."""
    # Initialize the object prior system
    object_prior = ObjectPrior()

    # Get the shape prior for the class
    prior_cloud = object_prior.get_shape_prior(class_name)
    if prior_cloud is None:
        print(f"No shape prior available for {class_name}")
        return mesh

    # If processing a tank and a standard model exists, use it
    if class_name.lower() == "tank" and object_prior.standard_tank_model is not None:
        prior_cloud = object_prior.standard_tank_model

    try:
        # Sample point cloud from the mesh
        if hasattr(mesh, 'sample_points_uniformly'):
            source_pcd = mesh.sample_points_uniformly(number_of_points=2000)
        else:
            # Manually sample point cloud
            points = np.asarray(mesh.vertices)
            if len(points) > 2000:
                indices = np.random.choice(len(points), 2000, replace=False)
                points = points[indices]
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(points)

        # Sample from the prior point cloud
        target_pcd = object_prior.sample_points(prior_cloud, 2000)

        # Calculate initial transformation (based on centroids)
        source_center = np.asarray(source_pcd.get_center())
        target_center = np.asarray(target_pcd.get_center())
        initial_translation = target_center - source_center

        # Create initial transformation matrix
        initial_transform = np.eye(4)
        initial_transform[:3, 3] = initial_translation

        # Execute registration
        reg_result = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, max_correspondence_distance=0.1,
            init=initial_transform,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=100,
                relative_fitness=1e-6,
                relative_rmse=1e-6
            )
        )

        # Apply transformation based on fit strength
        if fit_strength >= 1.0:
            # Full fit
            transformation = reg_result.transformation
        else:
            # Partial fit: interpolate transformation matrix
            identity = np.eye(4)
            transformation = object_prior.interpolate_transformation(identity, reg_result.transformation, fit_strength)

        # Apply transformation
        mesh.transform(transformation)
        print(
            f"Applying shape prior to {class_name} (Fit Strength: {fit_strength:.2f}, Fitness: {reg_result.fitness:.3f})")

    except Exception as e:
        print(f"Error applying shape prior: {str(e)}")

    return mesh
