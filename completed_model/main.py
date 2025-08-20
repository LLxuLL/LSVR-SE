import copy
import os
import shutil

import open3d as o3d
from datetime import datetime
import torch
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import KDTree

from edit_model import UniversalModelEditor, parse_edit_command
from window_editor import SemanticWindowEditor
from multiview_gan import generate_multiview
from normal_gray_generation import generate_normal_gray
from point_cloud_generation import generate_point_cloud
from point_cloud_processing import align_point_clouds, refine_point_cloud
from mesh_generation import create_mesh
import time
import traceback
import warnings
import logging
from object_prior import ObjectPrior


warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
warnings.filterwarnings("ignore", category=FutureWarning)


def generate_typical_shape(object_prior, class_name, output_dir):
    """Typical shapes for this category are generated directly from shape priors"""
    print(f"Generating typical shape for class: {class_name}")
    start_time = time.time()

    # Get the typical shapes of this category
    typical_pcd = object_prior.get_shape_prior(class_name)

    if typical_pcd is None:
        print(f"Failed to generate typical shape for {class_name}")
        return False

    # Create a grid
    typical_mesh = create_typical_mesh(typical_pcd)

    # Save typical shapes (PLY format preserves normal information)
    typical_mesh_path = os.path.join(output_dir, f"typical_{class_name}_shape.ply")
    o3d.io.write_triangle_mesh(typical_mesh_path, typical_mesh)
    print(f"  Saved typical shape to {typical_mesh_path}")
    print(f"  Completed in {time.time() - start_time:.2f} seconds")
    return True


def create_typical_mesh(pcd):
    """Create a typical mesh-shaped mesh from the point cloud"""

    if not pcd.has_normals():
        print("  Estimating normals for typical shape point cloud...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

    # Poisson reconstruction - adjust parameters
    print("  Running Poisson surface reconstruction...")
    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=10, linear_fit=True, n_threads=4
        )

        # Remove low-density vertices
        if len(densities) > 0:
            print("  Removing low-density vertices...")
            density_threshold = np.quantile(densities, 0.1)
            vertices_to_remove = densities < density_threshold
            mesh.remove_vertices_by_mask(vertices_to_remove)
    except Exception as e:
        print(f"  Poisson failed: {str(e)}, using Ball-Pivoting")
        radii = [0.005, 0.01, 0.02, 0.04]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )

    # Grid simplification
    print("  Simplifying mesh...")
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=30000)

    # Mesh smoothing
    print("  Smoothing mesh...")
    mesh = mesh.filter_smooth_taubin(number_of_iterations=10)

    # Calculates vertex normals
    print("  Computing vertex normals...")
    mesh.compute_vertex_normals()

    return mesh


def semantic_edit_mesh(mesh_path: str, edit_command: str, output_path: str, model_type: str) -> bool:
    """Semantic Editing Mesh - Use the Universal Model Editor"""
    try:

        editor = UniversalModelEditor("./models/model_configs")


        print(f"Try loading the grid: {mesh_path}")
        mesh = o3d.io.read_triangle_mesh(mesh_path)

        if not mesh.has_vertices():
            print(f"Error: Failed to load grid or grid is empty: {mesh_path}")
            return False

        # Make sure the grid has triangular patches
        if not mesh.has_triangles():
            print("Warning: The mesh does not have triangular patches, try to rebuild")

            # Reconstruct the mesh from the point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = mesh.vertices
            if mesh.has_vertex_normals():
                pcd.normals = mesh.vertex_normals
            else:
                pcd.estimate_normals()

            # Poisson reconstruction
            try:
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=8
                )
                # Remove low-density areas
                if len(densities) > 0:
                    vertices_to_remove = densities < np.quantile(densities, 0.01)
                    mesh.remove_vertices_by_mask(vertices_to_remove)
                print("Successfully reconstruct the mesh from the point cloud")
            except Exception as e:
                print(f"Poisson reconstruction failed: {e}, attempted convex package algorithm")
                mesh, _ = pcd.compute_convex_hull()

        # Make sure the mesh is normal
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()

        print(f"Mesh loads successfully: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

        # 解析编辑命令
        operation_name, parameters = parse_edit_command(edit_command, model_type)

        if not operation_name:
            print(f"Unable to parse edit commands: {edit_command}")
            return False

        print(f"Action: {operation_name}, Parameters: {parameters}")


        edited_mesh = editor.edit_model(model_type, operation_name, mesh, parameters)


        if edited_mesh is None or len(edited_mesh.vertices) == 0:
            print("The edited mesh is invalid, save the original mesh")
            edited_mesh = mesh


        os.makedirs(os.path.dirname(output_path), exist_ok=True)


        if not edited_mesh.has_triangles() or len(edited_mesh.triangles) == 0:
            print("The edited mesh does not have triangular patches and is saved as a point cloud")

            pcd = o3d.geometry.PointCloud()
            pcd.points = edited_mesh.vertices
            if edited_mesh.has_vertex_normals():
                pcd.normals = edited_mesh.vertex_normals
            o3d.io.write_point_cloud(output_path, pcd)
        else:

            o3d.io.write_triangle_mesh(output_path, edited_mesh)

        print(f"The semantic editing is complete, and the result is saved to: {output_path}")

        return True

    except Exception as e:
        print(f"Semantic editing failed: {str(e)}")
        traceback.print_exc()
        return False


def visualize_point_clouds(pcd1, pcd2, label1, label2, save_path, num_points=1000):
    """
    Visualize two point clouds and save the contrasting image
    """
    # Downsample point clouds to improve visualization efficiency
    def downsample_pcd(pcd, num_points):
        if len(pcd.points) > num_points:
            indices = np.random.choice(len(pcd.points), num_points, replace=False)
            return np.asarray(pcd.points)[indices]
        return np.asarray(pcd.points)

    points1 = downsample_pcd(pcd1, num_points)
    points2 = downsample_pcd(pcd2, num_points)

    fig = plt.figure(figsize=(12, 6))

    # The first subgraph: point cloud 1
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(points1[:, 0], points1[:, 1], points1[:, 2], s=1, alpha=0.5)
    ax1.set_title(label1)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # The second sub-image: Point Cloud 2
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(points2[:, 0], points2[:, 1], points2[:, 2], s=1, alpha=0.5)
    ax2.set_title(label2)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Save the point cloud comparison map to: {save_path}")

def main(input_image_path, fit_strength=0.9 ,edit_command: str = None):
    # Create an output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./output/output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    latest_dir = "./output/output_latest"
    os.makedirs(latest_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=os.path.join(output_dir, 'debug.log'),
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    # Log the processing log
    log_file = os.path.join(output_dir, "processing_log.txt")
    with open(log_file, "w") as f:
        f.write(f"Processing started at: {datetime.now()}\n")
        f.write(f"Input image: {input_image_path}\n")

    # Initialize the object prior knowledge system
    object_prior = ObjectPrior(device='cuda' if torch.cuda.is_available() else 'cpu')

    # Detect object categories
    class_name, class_idx = object_prior.detect_object_class(input_image_path)
    print(f"Detected object class: {class_name} (index: {class_idx})")
    with open(log_file, "a") as f:
        f.write(f"Detected object class: {class_name} (index: {class_idx})\n")

    # Step 0: Generate category typical shapes directly
    print(f"Step 0/6: Generating typical shape for {class_name}...")
    with open(log_file, "a") as f:
        f.write(f"Step 0/6: Generating typical shape for {class_name}...\n")

    if generate_typical_shape(object_prior, class_name, output_dir):
        print("  Typical shape generation successful")
        with open(log_file, "a") as f:
            f.write("  Typical shape generation successful\n")
    else:
        print("  Typical shape generation failed")
        with open(log_file, "a") as f:
            f.write("  Typical shape generation failed\n")

    try:
        print(f"Step 1/6: Generating multi-view images from {input_image_path}...")
        with open(log_file, "a") as f:
            f.write(f"Step 1/6: Generating multi-view images from {input_image_path}...\n")

        start_time = time.time()


        multiview_images = generate_multiview(
            input_image_path,
            angles=["front", "back", "side_one", "side_two", "top", "bottom"]
        )

        elapsed = time.time() - start_time
        print(f"  Completed in {elapsed:.2f} seconds")
        with open(log_file, "a") as f:
            f.write(f"  Completed in {elapsed:.2f} seconds\n")


        for angle, img in multiview_images.items():
            img_path = os.path.join(output_dir, f"{angle}_view.png")
            img.save(img_path)
            print(f"  Saved {angle} view to {img_path}")
            with open(log_file, "a") as f:
                f.write(f"  Saved {angle} view to {img_path}\n")

        print("Step 2/6: Generating normal and depth maps...")
        with open(log_file, "a") as f:
            f.write("Step 2/6: Generating normal and depth maps...\n")

        start_time = time.time()
        normal_gray_data = generate_normal_gray(multiview_images)

        elapsed = time.time() - start_time
        print(f"  Completed in {elapsed:.2f} seconds")
        with open(log_file, "a") as f:
            f.write(f"  Completed in {elapsed:.2f} seconds\n")


        for angle, data in normal_gray_data.items():
            normal_path = os.path.join(output_dir, f"{angle}_normal.png")
            data["normal"].save(normal_path)
            print(f"  Saved {angle} normal map to {normal_path}")
            with open(log_file, "a") as f:
                f.write(f"  Saved {angle} normal map to {normal_path}\n")

        print("Step 3/6: Generating point clouds...")
        with open(log_file, "a") as f:
            f.write("Step 3/6: Generating point clouds...\n")

        start_time = time.time()
        point_clouds = generate_point_cloud(normal_gray_data)

        elapsed = time.time() - start_time
        print(f"  Completed in {elapsed:.2f} seconds")
        with open(log_file, "a") as f:
            f.write(f"  Completed in {elapsed:.2f} seconds\n")


        for angle, pcd in point_clouds.items():
            pcd_path = os.path.join(output_dir, f"{angle}_cloud.ply")
            o3d.io.write_point_cloud(pcd_path, pcd)
            print(f"  Saved {angle} point cloud to {pcd_path}")
            with open(log_file, "a") as f:
                f.write(f"  Saved {angle} point cloud to {pcd_path}\n")

        print(f"Step 4/6: Aligning and refining point clouds (fit: {fit_strength:.2f})...")
        with open(log_file, "a") as f:
            f.write(f"Step 4/6: Aligning and refining point clouds (fit: {fit_strength:.2f})...\n")

        start_time = time.time()
        combined_pcd = align_point_clouds(point_clouds, class_name=class_name, fit_strength=fit_strength)
        refined_pcd = refine_point_cloud(combined_pcd)


        aligned_path = os.path.join(output_dir, "aligned.ply")
        o3d.io.write_point_cloud(aligned_path, refined_pcd)


        visualize_path = os.path.join(output_dir, "point_cloud_aligned.png")
        visualize_point_clouds(combined_pcd, refined_pcd, "The original point cloud", "Optimize the post-point cloud", visualize_path)

        elapsed = time.time() - start_time
        print(f"Point cloud alignment completes in the following time: {elapsed:.2f} seconds")

        print("Step 5/6: Creating reconstructed mesh...")
        with open(log_file, "a") as f:
            f.write("Step 5/6: Creating reconstructed mesh...\n")

        start_time = time.time()
        # Pass the object category information and fit strength to the mesh generation
        mesh = create_mesh(refined_pcd, class_name=class_name, fit_strength=fit_strength)


        pre_fit_path = os.path.join(output_dir, "pre_fit.ply")
        o3d.io.write_point_cloud(pre_fit_path, refined_pcd)


        if fit_strength > 0:

            object_prior = ObjectPrior()
            fitted_pcd = object_prior.apply_shape_prior(refined_pcd, class_name, fit_strength)


            post_fit_path = os.path.join(output_dir, "post_fit.ply")
            o3d.io.write_point_cloud(post_fit_path, fitted_pcd)


            fit_compare_path = os.path.join(output_dir, "point_cloud_fit_comparison.png")
            visualize_point_clouds(refined_pcd, fitted_pcd, "It fits the previous point", f"Post-Fitting Point Cloud (Intensity = {fit_strength})",
                                   fit_compare_path)

        elapsed = time.time() - start_time
        print(f"The grid is generated in the following time: {elapsed:.2f} seconds")

        post_fit_path = os.path.join(output_dir, "post_fit.ply")
        o3d.io.write_point_cloud(post_fit_path, fitted_pcd)
        print(f"Save the fitted point cloud to {post_fit_path}")


        latest_model_path = os.path.join(latest_dir, f"{class_name}.ply")
        os.makedirs(os.path.dirname(latest_model_path), exist_ok=True)
        shutil.copy(post_fit_path, latest_model_path)
        print(f"Copy fit point cloud to {latest_model_path}")

        # Also copy to edited_input directory for subsequent editing
        edited_input_path = os.path.join("./output/edited_input", f"{class_name}.ply")
        os.makedirs(os.path.dirname(edited_input_path), exist_ok=True)
        shutil.copy(post_fit_path, edited_input_path)
        print(f"Copy-fitting point clouds to {edited_input_path}")


        print("Step 6/6: Creating reconstructed mesh...")
        with open(log_file, "a") as f:
            f.write("Step 6/6: Creating reconstructed mesh...\n")

        start_time = time.time()

        mesh = create_mesh(fitted_pcd)

        elapsed = time.time() - start_time
        print(f"网格生成完成，耗时: {elapsed:.2f}秒")


        mesh_path = os.path.join(output_dir, "reconstructed_mesh.ply")
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        print(f"Save the rebuild mesh to {mesh_path}")


        for angle in ["front", "back", "side_one", "side_two", "top", "bottom"]:

            img_src = os.path.join(output_dir, f"{angle}_view.png")
            img_dst = os.path.join(latest_dir, f"{angle}_view.png")
            if os.path.exists(img_src):
                shutil.copy(img_src, img_dst)
                print(f"Copy {angle} view to {img_dst}")
            else:
                print(f"Warning: {img_src} not found")


            normal_src = os.path.join(output_dir, f"{angle}_normal.png")
            normal_dst = os.path.join(latest_dir, f"{angle}_normal.png")
            if os.path.exists(normal_src):
                shutil.copy(normal_src, normal_dst)
                print(f"Copy {angle} normal map to {normal_dst}")
            else:
                print(f"Warning: {normal_src} not found")

        print("Done! The results are saved at:", output_dir)
        return True, class_name

    except Exception as e:
        error_msg = f"\nAn error occurred during processing: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        with open(log_file, "a") as f:
            f.write(error_msg)
        return False, ""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='3D reconstruction system')
    parser.add_argument('--input', type=str, default="./test_image/1.png", help='Enter the image path')
    parser.add_argument('--fit', type=float, default=1, help='Fit Strength (0.0-1.0)')
    args = parser.parse_args()


    if not os.path.exists(args.input):
        print(f"Error: The input image does not exist in {args.input}")
    else:
        print(f"Start processing... Fit Strength: {args.fit:.2f}")
        success, class_name = main(args.input, fit_strength=args.fit)
        if not success:
            print("Processing failed")