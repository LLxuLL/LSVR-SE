import copy
import heapq
import json
import time

import torch
import numpy as np
import open3d as o3d
import os

import trimesh
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from torch import cdist
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torchvision.models import vgg16
import scipy.spatial.transform as sst
from scipy.spatial import KDTree, ConvexHull

print(torch.cuda.is_available())     # True
print(torch.cuda.get_device_name(0)) # NVIDIA GeForce RTX 3080

# Define CIFAR-100 class list (for classifier)
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'window',
    'woman', 'worm'
]


class PointNetEncoder(nn.Module):
    """Point Cloud Encoder"""

    def __init__(self, latent_dim=128):
        super(PointNetEncoder, self).__init__()
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, latent_dim, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(latent_dim)

        self.fc = nn.Linear(latent_dim, latent_dim)
        self.bn4 = nn.BatchNorm1d(latent_dim)

    def forward(self, x):
        # x: (batch_size, num_points, 3)
        x = x.transpose(1, 2)  # (batch_size, 3, num_points)

        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))

        # Global features
        x = torch.max(x, 2)[0]  # (batch_size, latent_dim)

        # Fully connected layer
        x = torch.relu(self.bn4(self.fc(x)))

        return x


class ShapePriorDecoder(nn.Module):
    """Shape Prior Decoder"""

    def __init__(self, latent_dim=128, num_points=2048):
        super(ShapePriorDecoder, self).__init__()
        self.num_points = num_points

        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, num_points * 3)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)

        # Reshape to point cloud
        x = x.view(-1, self.num_points, 3)
        return x


class ShapePriorModel(nn.Module):
    """Complete Shape Prior Model"""

    def __init__(self, latent_dim=128, num_points=2048):
        super(ShapePriorModel, self).__init__()
        self.encoder = PointNetEncoder(latent_dim)
        self.decoder = ShapePriorDecoder(latent_dim, num_points)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

class ObjectPrior:
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        self.classifier = None
        self.shape_prior_model = None
        self.class_encoder = LabelEncoder()
        self.loaded_classes = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.standard_models = {}  # Store standard models for each class
        self.standard_kdtrees = {}  # Store KDTree for each class

        # Load models
        self.load_models()

    def load_standard_model(self, class_name):
        """Load standard model for specified class and build KDTree"""
        # Check if already loaded
        if class_name in self.standard_models:
            return self.standard_models[class_name]

        # Updated to use class-named PLY format
        standard_path = f"./models/standard_models/standard_model_{class_name}.ply"

        if os.path.exists(standard_path):
            try:
                # Directly load point cloud
                model = o3d.io.read_point_cloud(standard_path)

                if not model.is_empty():
                    # Estimate normals
                    model.estimate_normals(
                        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
                    )
                    # Build KDTree for nearest point search
                    points = np.asarray(model.points)
                    kdtree = KDTree(points)

                    # Store in dictionary
                    self.standard_models[class_name] = model
                    self.standard_kdtrees[class_name] = kdtree
                    print(f"Successfully loaded {class_name} standard model and built KDTree")
                    return model
                else:
                    print(f"{class_name} standard model is empty")
            except Exception as e:
                print(f"Failed to load {class_name} standard model: {str(e)}")
        else:
            print(f"Standard model file does not exist: {standard_path}")

        return None

    def apply_point_based_fitting(self, point_cloud, class_name, fit_strength=0.9, iterations=3):
        """Improved point-to-point fitting method - using standard model of specified class"""
        # Load standard model for specified class
        standard_model = self.load_standard_model(class_name)
        if standard_model is None or class_name not in self.standard_kdtrees:
            print(f"Cannot apply point-to-point fitting: {class_name} standard model not loaded")
            return point_cloud

        std_kdtree = self.standard_kdtrees[class_name]
        standard_points = np.asarray(standard_model.points)
        num_standard = len(standard_points)

        # Create copy of point cloud
        working_cloud = copy.deepcopy(point_cloud)
        points = np.asarray(working_cloud.points)
        num_points = len(points)

        print(f"Input point cloud: {num_points} points, {class_name} standard point cloud: {num_standard} points")

        # Point cloud sampling: limit maximum points
        MAX_POINTS = 3000
        if num_points > MAX_POINTS:
            working_cloud = working_cloud.farthest_point_down_sample(MAX_POINTS)
            points = np.asarray(working_cloud.points)
            num_points = len(points)
            print(f"Input point cloud downsampled to {num_points} points")

        # Initialize point status array (0=not fitted, 1=fitted)
        point_status = np.zeros(num_points, dtype=int)

        # Create KDTree for standard point cloud
        std_kdtree = KDTree(standard_points)

        # Multiple fitting iterations
        for iter in range(iterations):
            # 1. Mark unfitted points
            unmatched_indices = np.where(point_status == 0)[0]
            print(f"Iteration {iter + 1}: Found {len(unmatched_indices)} unfitted points")

            # 2. Find nearest standard points for unfitted points
            if len(unmatched_indices) > 0:
                unmatched_points = points[unmatched_indices]
                distances, std_indices = std_kdtree.query(unmatched_points, k=1)

                # 3. Apply displacement vectors
                displacement_vectors = standard_points[std_indices] - unmatched_points
                unmatched_points += displacement_vectors * fit_strength

                # Update points in point cloud
                points[unmatched_indices] = unmatched_points

                # 4. Mark fitted points
                point_status[unmatched_indices] = 1

            # 5. Update point cloud and normals
            working_cloud.points = o3d.utility.Vector3dVector(points)
            working_cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # 6. Point cloud fusion: add unmatched points from standard model
        matched_std_indices = set()
        distances, _ = std_kdtree.query(points, k=1)
        matched_std_indices = set(std_indices)

        unmatched_std_indices = set(range(num_standard)) - matched_std_indices
        if unmatched_std_indices:
            print(f"Adding {len(unmatched_std_indices)} unmatched standard points")
            new_points = np.vstack([points, standard_points[list(unmatched_std_indices)]])
            working_cloud.points = o3d.utility.Vector3dVector(new_points)
            working_cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        return working_cloud

    def upsample_point_cloud(self, pcd, target_num_points):
        """Point cloud upsampling to target number of points (efficient version)"""
        # 1. Get original points
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals) if pcd.has_normals() else None

        # 2. Calculate number of points needed
        num_needed = max(0, target_num_points - len(points))
        if num_needed == 0:
            return pcd

        # 3. Simple and efficient upsampling method
        new_points = []
        new_normals = [] if normals is not None else None

        # Calculate point cloud boundaries
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)
        size = max_vals - min_vals

        # Generate random points within boundaries
        for _ in range(num_needed):
            rand_point = min_vals + np.random.random(3) * size
            new_points.append(rand_point)

        # 4. Create new point cloud
        all_points = np.vstack([points, np.array(new_points)])
        upsampled_pcd = o3d.geometry.PointCloud()
        upsampled_pcd.points = o3d.utility.Vector3dVector(all_points)

        if new_normals is not None and normals is not None:
            # Calculate normals for new points (using nearest point normals)
            kdtree = KDTree(points)
            _, indices = kdtree.query(np.array(new_points))
            all_normals = np.vstack([normals, normals[indices]])
            upsampled_pcd.normals = o3d.utility.Vector3dVector(all_normals)

        return upsampled_pcd

    def estimate_point_curvature(self, pcd, radius=0.05):
        """Estimate curvature of points in point cloud (for upsampling weights)"""
        points = np.asarray(pcd.points)
        curvatures = np.zeros(len(points))

        # Build KDTree for nearest neighbor search
        kdtree = KDTree(points)

        for i in range(len(points)):
            # Find neighbors within radius
            indices = kdtree.query_ball_point(points[i], radius)
            neighbors = points[indices]

            if len(neighbors) < 5:
                curvatures[i] = 0.1  # Default value
                continue

            # Calculate covariance matrix
            cov_matrix = np.cov(neighbors.T)

            # Calculate eigenvalues
            eigenvalues = np.linalg.eigvalsh(cov_matrix)
            eigenvalues.sort()  # Ascending order

            # Curvature estimation: smallest eigenvalue / sum of eigenvalues
            curvatures[i] = eigenvalues[0] / (np.sum(eigenvalues) + 1e-8)

        # Normalize curvature values
        curvatures = (curvatures - np.min(curvatures)) / (np.max(curvatures) - np.min(curvatures) + 1e-8)
        return curvatures

    def generate_point_in_convex_hull(self, points, min_vals, max_vals):
        """Generate random point within convex hull of point cloud"""
        # Calculate convex hull
        hull = ConvexHull(points)

        # Generate points within convex hull
        while True:
            # Calculate random combination of convex hull vertices
            coeffs = np.random.random(len(hull.vertices))
            coeffs /= coeffs.sum()

            # Generate point within convex hull
            new_point = np.dot(coeffs, points[hull.vertices])

            # Check if within boundaries
            if np.all(new_point >= min_vals) and np.all(new_point <= max_vals):
                return new_point

    def apply_shape_prior(self, point_cloud, class_name="window", fit_strength=0.9):
        """Apply shape prior - using standard model of specified class"""
        # Only apply fitting to supported classes
        if fit_strength <= 0:
            return point_cloud

        # Try to load standard model for specified class
        if class_name.lower() not in self.standard_models:
            self.load_standard_model(class_name.lower())

        # Apply point-to-point fitting if standard model exists for this class
        if class_name.lower() in self.standard_models:
            print(f"Applying point-to-point fitting (Class: {class_name}, Strength: {fit_strength:.2f})")
            return self.apply_point_based_fitting(point_cloud, class_name.lower(), fit_strength)

        return point_cloud

    def visualize_matching(self, original_points, matched_points, save_path):
        """Visualize matching results"""
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot original point cloud (blue)
        ax.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2],
                   c='blue', s=5, alpha=0.3, label='Original Points')

        # Plot matched point cloud (red)
        ax.scatter(matched_points[:, 0], matched_points[:, 1], matched_points[:, 2],
                   c='red', s=5, alpha=0.5, label='Matched Points')

        # Plot standard point cloud (green)
        ax.scatter(self.standard_points[:, 0], self.standard_points[:, 1], self.standard_points[:, 2],
                   c='green', s=10, alpha=0.2, label='Standard Points')

        ax.set_title('Point Cloud Matching Results')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved matching visualization to: {save_path}")

    def sample_points(self, pcd, num_points):
        """More compatible point cloud sampling method"""
        # Manual implementation of uniform sampling
        points = np.asarray(pcd.points)
        if len(points) <= num_points:
            return pcd

        indices = np.random.choice(len(points), num_points, replace=False)
        sampled_pcd = o3d.geometry.PointCloud()
        sampled_pcd.points = o3d.utility.Vector3dVector(points[indices])

        if pcd.has_normals():
            normals = np.asarray(pcd.normals)[indices]
            sampled_pcd.normals = o3d.utility.Vector3dVector(normals)

        if pcd.has_colors():
            colors = np.asarray(pcd.colors)[indices]
            sampled_pcd.colors = o3d.utility.Vector3dVector(colors)

        return sampled_pcd

    def interpolate_transformation(self, T1, T2, alpha):
        """Interpolate between two transformation matrices - corrected version"""
        # Decompose transformation matrices
        R1 = T1[:3, :3]
        t1 = T1[:3, 3]
        R2 = T2[:3, :3]
        t2 = T2[:3, 3]

        # Interpolate rotation matrices (using quaternion spherical interpolation)
        rot1 = sst.Rotation.from_matrix(R1)
        rot2 = sst.Rotation.from_matrix(R2)
        R_interp = (rot1 * sst.Rotation.from_rotvec(rot2.as_rotvec() * alpha)).as_matrix()

        # Interpolate translation vectors
        t_interp = (1 - alpha) * t1 + alpha * t2

        # Combine new transformation matrix
        T_interp = np.eye(4)
        T_interp[:3, :3] = R_interp
        T_interp[:3, 3] = t_interp

        return T_interp

    def slerp_rotation(self, R1, R2, alpha):
        """Spherical linear interpolation of rotation matrices"""
        # Convert to quaternions
        q1 = self.rotation_matrix_to_quaternion(R1)
        q2 = self.rotation_matrix_to_quaternion(R2)

        # Quaternion interpolation
        q_interp = self.quaternion_slerp(q1, q2, alpha)

        # Convert back to rotation matrix
        return self.quaternion_to_rotation_matrix(q_interp)

    def rotation_matrix_to_quaternion(self, R):
        """Rotation matrix to quaternion - using scipy conversion"""
        # Use scipy's Rotation class
        rotation = sst.Rotation.from_matrix(R)
        return rotation.as_quat()

    def quaternion_to_rotation_matrix(self, q):
        """Quaternion to rotation matrix - using scipy conversion"""
        # Use scipy's Rotation class
        rotation = sst.Rotation.from_quat(q)
        return rotation.as_matrix()

    def quaternion_slerp(self, q1, q2, alpha):
        """Quaternion spherical linear interpolation - using scipy slerp"""
        # Create Rotation objects
        rot1 = sst.Rotation.from_quat(q1)
        rot2 = sst.Rotation.from_quat(q2)

        # Perform slerp interpolation
        interpolated = sst.Rotation.from_rotvec(rot1.as_rotvec() * (1 - alpha) + rot2.as_rotvec() * alpha)
        return interpolated.as_quat()

    def load_models(self):
        """Load pre-trained models"""
        # Classifier path
        classifier_path = "./classifier_models/vgg16_cifar100_final.pth"

        # Shape prior path
        shape_prior_path = "./output/shape_prior_models/shape_prior_final.pth"
        class_list_path = "./data/shape_prior_dataset/full/class_list.txt"
        class_mapping_path = "./data/shape_prior_dataset/full/class_mapping.json"

        # Load classifier
        if os.path.exists(classifier_path):
            self.classifier = vgg16(pretrained=False)
            num_features = self.classifier.classifier[6].in_features
            self.classifier.classifier[6] = nn.Linear(num_features, 100)
            self.classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))
            self.classifier = self.classifier.to(self.device).eval()
            print("Loaded object classifier")
            # Use CIFAR-100 classes as default
            self.loaded_classes = CIFAR100_CLASSES
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.loaded_classes)}
            self.idx_to_class = {idx: cls for idx, cls in enumerate(self.loaded_classes)}
        else:
            print("Warning: Object classifier not found. Using CIFAR-100 classes as fallback.")
            self.loaded_classes = CIFAR100_CLASSES
            self.class_encoder.fit(self.loaded_classes)
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.loaded_classes)}
            self.idx_to_class = {idx: cls for idx, cls in enumerate(self.loaded_classes)}

        # Load shape prior model
        if os.path.exists(shape_prior_path):
            self.shape_prior_model = ShapePriorModel(latent_dim=128, num_points=2048)
            self.shape_prior_model.load_state_dict(torch.load(shape_prior_path, map_location=self.device))
            self.shape_prior_model = self.shape_prior_model.to(self.device).eval()
            print("Loaded shape prior model")

            # Load class list
            if os.path.exists(class_list_path):
                with open(class_list_path, 'r') as f:
                    self.loaded_classes = [line.strip() for line in f.readlines()]
                print(f"Loaded {len(self.loaded_classes)} classes")

                # Load class mapping
                if os.path.exists(class_mapping_path):
                    with open(class_mapping_path, 'r') as f:
                        self.class_to_idx = json.load(f)
                    # Create reverse mapping
                    self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
                    print(f"Loaded class mapping with {len(self.class_to_idx)} entries")
                else:
                    print("Warning: Class mapping not found. Creating default mapping.")
                    self.class_to_idx = {cls: idx for idx, cls in enumerate(self.loaded_classes)}
                    self.idx_to_class = {idx: cls for idx, cls in enumerate(self.loaded_classes)}
            else:
                print("Warning: Class list not found. Using classifier classes.")
        else:
            print("Warning: Shape prior model not found. Continuing without shape prior.")

    def detect_object_class(self, image_path):
        """Detect object class in image"""
        if self.classifier is None:
            return "window", 0

        # Image preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(self.device)

        # Predict class
        with torch.no_grad():
            outputs = self.classifier(image_tensor)
            _, predicted = outputs.max(1)
            class_idx = predicted.item()

            # Get class name
            class_name = CIFAR100_CLASSES[class_idx] if class_idx < len(CIFAR100_CLASSES) else "unknown"

            return class_name, class_idx

    def get_shape_prior(self, class_name):
        """Get typical shape for specific class - improved version"""
        # Check if class exists
        if class_name not in self.class_to_idx:
            print(f"No shape prior available for '{class_name}'. Available classes: {list(self.class_to_idx.keys())}")
            return None

        try:
            # Get class index
            class_idx = self.class_to_idx[class_name]
            print(f"Generating shape prior for '{class_name}' (index: {class_idx})")

            # Create class-specific latent vector
            latent_vector = torch.zeros(1, 128).to(self.device)

            # Add class information (first 10 dimensions)
            for i in range(min(10, len(self.class_to_idx))):
                latent_vector[0, i] = class_idx / len(self.class_to_idx)

            # Add random variation (last 118 dimensions)
            latent_vector[0, 10:] = torch.randn(118) * 0.1

            # Decode to point cloud
            with torch.no_grad():
                prior_points = self.shape_prior_model.decoder(latent_vector)

            # Create point cloud object
            prior_cloud = o3d.geometry.PointCloud()
            prior_points = prior_points.squeeze(0).cpu().numpy()
            prior_cloud.points = o3d.utility.Vector3dVector(prior_points)

            # Apply class-specific transformation
            self.apply_class_specific_transform(prior_cloud, class_name)

            # Estimate normals
            if not prior_cloud.has_normals():
                prior_cloud.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
                )

            return prior_cloud

        except Exception as e:
            print(f"Error generating shape prior for '{class_name}': {str(e)}")
            return None

    def apply_class_specific_transform(self, pcd, class_name):
        """Apply class-specific transformation - improved version"""
        # Tank class specific optimization
        if class_name.lower() == "tank":
            # Normalize point cloud
            points = np.asarray(pcd.points)
            centroid = np.mean(points, axis=0)
            points -= centroid
            max_distance = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
            if max_distance > 0:
                points /= max_distance

            # Apply specific scaling
            points[:, 0] *= 1.5  # X-axis scaling (body length)
            points[:, 1] *= 0.8  # Y-axis scaling (height)
            points[:, 2] *= 0.7  # Z-axis scaling (width)

            # Add tank features
            # 1. Create chassis
            base_points = np.copy(points)
            base_points[:, 1] = base_points[:, 1].min() - 0.1
            base_points[:, 0] *= 1.2
            base_points[:, 2] *= 1.2

            # 2. Create turret
            turret_mask = (points[:, 0] > 0.3) & (np.abs(points[:, 2]) < 0.2)
            turret_points = np.copy(points[turret_mask])
            turret_points[:, 1] += 0.2

            # Combine points
            all_points = np.vstack([points, base_points, turret_points])
            pcd.points = o3d.utility.Vector3dVector(all_points)

        else:
            # Normalize point cloud
            points = np.asarray(pcd.points)
            centroid = np.mean(points, axis=0)
            points -= centroid
            max_distance = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
            if max_distance > 0:
                points /= max_distance
            pcd.points = o3d.utility.Vector3dVector(points)