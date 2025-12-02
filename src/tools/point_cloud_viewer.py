import open3d as o3d
import numpy as np
import argparse


def visualize_point_cloud(file_path):
    """
    Load and visualize point cloud file (supports PLY and OBJ formats)

    Parameters:
        file_path (str): Point cloud file path
    """
    # Check file extension
    if file_path.lower().endswith('.ply'):
        # Read PLY file
        pcd = o3d.io.read_point_cloud(file_path)
    elif file_path.lower().endswith('.obj'):
        # Read OBJ file and convert to point cloud
        mesh = o3d.io.read_triangle_mesh(file_path)
        pcd = mesh.sample_points_uniformly(number_of_points=50000)  # Uniformly sample points from mesh
    else:
        raise ValueError("Unsupported file format, please provide PLY or OBJ file")

    # Check if point cloud is empty
    if not pcd.has_points():
        raise ValueError("Unable to load point cloud data, file may be empty or format incorrect")

    # Print point cloud information
    print("Point cloud file:", file_path)
    print("Number of points:", np.asarray(pcd.points).shape[0])
    print("Bounding box size:", pcd.get_axis_aligned_bounding_box().get_extent())

    # Optional: Point cloud preprocessing
    # 1. Downsample (if too many points)
    if np.asarray(pcd.points).shape[0] > 200000:
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        print("Number of points after downsampling:", np.asarray(pcd.points).shape[0])

    # 2. Estimate normals (for better rendering effect)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))

    # Set visualization parameters
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name='3D Point Cloud Viewer - ' + file_path, width=1200, height=800)

    # Add point cloud to visualizer
    visualizer.add_geometry(pcd)

    # Set rendering options
    render_opt = visualizer.get_render_option()
    render_opt.point_size = 2.0  # Point size
    render_opt.background_color = np.array([0.1, 0.1, 0.1])  # Background color
    render_opt.light_on = True  # Enable lighting

    # Set view control
    view_ctl = visualizer.get_view_control()
    view_ctl.set_zoom(0.8)  # Initial zoom

    print("\nControl instructions:")
    print(" - Left mouse button: Rotate view")
    print(" - Right mouse button: Pan view")
    print(" - Mouse wheel: Zoom")
    print(" - R: Reset view")
    print(" - C: Toggle point/wireframe rendering mode")
    print(" - Esc: Exit viewer")

    # Run visualization
    visualizer.run()
    visualizer.destroy_window()


if __name__ == "__main__":
    # Set command line argument parsing
    parser = argparse.ArgumentParser(description='3D Point Cloud Viewer')
    parser.add_argument('file_path', type=str, help='PLY or OBJ file path')
    args = parser.parse_args()

    # Run visualization
    visualize_point_cloud(args.file_path)