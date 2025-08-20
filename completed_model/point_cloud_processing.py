import copy
import numpy as np
import open3d as o3d
from scipy.ndimage import gaussian_filter


def align_point_clouds(point_clouds, class_name="unknown", fit_strength=0.9):
    """Align point clouds from multiple views - optimized version"""
    # Use front point cloud as reference
    if "front" in point_clouds:
        base_cloud = point_clouds["front"]
    else:
        base_cloud = next(iter(point_clouds.values()))

    combined = copy.deepcopy(base_cloud)

    # Define more precise initial transformation matrices
    transformations = {
        "front": np.eye(4),
        "side_one": np.array([
            [0.0, 0.0, -1.0, 0.5],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]),
        "side_two": np.array([
            [0.0, 0.0, 1.0, -0.5],
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]),
        "back": np.array([
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]),
        "top": np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.5],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]),
        "bottom": np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -0.5],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
    }

    # Apply transformations and merge point clouds
    for angle, cloud in point_clouds.items():
        if angle == "front":
            continue

        # Create copy of point cloud
        cloud_copy = copy.deepcopy(cloud)

        # Apply predefined transformation
        if angle in transformations:
            cloud_copy.transform(transformations[angle])

        # Ensure point cloud has normal information
        if not cloud_copy.has_normals():
            cloud_copy.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # Ensure reference point cloud has normal information
        if not base_cloud.has_normals():
            base_cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # Use ICP for fine registration
        try:
            # Downsample to improve registration efficiency
            cloud_down = cloud_copy.voxel_down_sample(voxel_size=0.01)
            base_down = base_cloud.voxel_down_sample(voxel_size=0.01)

            # Perform point-to-plane ICP registration
            icp_result = o3d.pipelines.registration.registration_icp(
                cloud_down, base_down, 0.05,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=100,
                    relative_fitness=1e-6,
                    relative_rmse=1e-6
                )
            )

            # Apply transformation to original point cloud
            cloud_copy.transform(icp_result.transformation)
        except Exception as e:
            print(f"ICP failed for {angle}: {str(e)}")

        combined += cloud_copy

    return combined


def refine_point_cloud(pcd):
    """Optimize point cloud - add point-to-point fitting"""
    # Downsample
    pcd = pcd.voxel_down_sample(voxel_size=0.005)

    # Remove statistical outliers
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = cl

    # Gaussian smoothing
    points = np.asarray(pcd.points)
    smoothed_points = np.zeros_like(points)

    # Apply Gaussian filter to each coordinate dimension
    for i in range(3):
        smoothed_points[:, i] = gaussian_filter(points[:, i], sigma=1.5)

    pcd.points = o3d.utility.Vector3dVector(smoothed_points)

    # Recalculate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))

    return pcd

def preprocess_point_cloud(pcd, voxel_size=0.05):
    """Point cloud preprocessing"""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size=0.05):
    """Execute global registration"""
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result