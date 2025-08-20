import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
from tqdm import tqdm
import argparse
import logging
import time
import random
import matplotlib.pyplot as plt
from matplotlib import cm
print(torch.cuda.is_available())     # True
print(torch.cuda.get_device_name(0)) # NVIDIA GeForce RTX 3080

# Disable unnecessary logging
logging.basicConfig(level=logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# All supported classes
ALL_CLASSES = [
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
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]


# Configuration parameters
class Config:
    def __init__(self, class_name):
        self.data_root = "./data/shape_prior_dataset/point_cloud"
        self.class_name = class_name
        self.num_points = 2048  # Reduce points to lower complexity
        self.batch_size = 32  # Small batch processing
        self.latent_dim = 128  # Noise vector dimension
        self.g_lr = 0.0001  # Learning rate
        self.d_lr = 0.0001  # Learning rate
        self.epochs = 1000  # Reduce number of epochs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create output directory
        self.sample_dir = f"./generated/{class_name}"
        os.makedirs(self.sample_dir, exist_ok=True)


# Point cloud dataset class
class PointCloudDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.file_list = self._get_file_list()

        if not self.file_list:
            raise ValueError(f"No PLY files found for class: {config.class_name}")

        self.point_clouds = self._load_and_preprocess()

    def _get_file_list(self):
        class_dir = os.path.join(self.config.data_root, self.config.class_name)
        if not os.path.exists(class_dir):
            return []
        return [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith('.ply')]

    def _load_ply(self, file_path):
        # Use open3d to load point cloud
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points, dtype=np.float32)
        return points

    def _resample_points(self, points, n):
        if len(points) == 0:
            return np.zeros((n, 3), dtype=np.float32)

        if len(points) > n:
            indices = np.random.choice(len(points), n, replace=False)
        else:
            indices = np.random.choice(len(points), n, replace=True)
        return points[indices]

    def _normalize_points(self, points):
        """Point cloud normalization"""
        centroid = np.mean(points, axis=0)
        points -= centroid
        max_dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        if max_dist > 1e-8:
            points /= max_dist
        return points

    def _load_and_preprocess(self):
        all_points = []
        for file_path in self.file_list:
            points = self._load_ply(file_path)
            points = self._resample_points(points, self.config.num_points)
            points = self._normalize_points(points)
            all_points.append(points)

        return np.array(all_points, dtype=np.float32)

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        return self.point_clouds[idx]


# Generator network - use MLP structure to avoid view issues
class Generator(nn.Module):
    def __init__(self, latent_dim, num_points):
        super().__init__()
        self.num_points = num_points

        # Simple MLP structure
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, num_points * 3)
        )

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(num_points * 3, num_points * 3),
            nn.Tanh()
        )

    def forward(self, z):
        # Generate point cloud data
        x = self.net(z)
        x = self.output_layer(x)
        # Directly reshape to point cloud format
        return x.view(-1, self.num_points, 3)


# Discriminator network - use global feature extraction
class Discriminator(nn.Module):
    def __init__(self, num_points):
        super().__init__()

        # Feature extraction layers
        self.feature_net = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.ReLU()
        )

        # Global max pooling
        self.pool = nn.AdaptiveMaxPool1d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Transpose point cloud data
        x = x.transpose(1, 2)
        # Extract features
        features = self.feature_net(x)
        # Global pooling
        global_features = self.pool(features).squeeze(2)
        # Classification
        return self.classifier(global_features)


# Point cloud GAN trainer - use standard GAN loss
class PointCloudGANTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device

        # Initialize networks
        self.G = Generator(config.latent_dim, config.num_points).to(self.device)
        self.D = Discriminator(config.num_points).to(self.device)

        # Optimizers
        self.opt_G = optim.Adam(self.G.parameters(), lr=config.g_lr)
        self.opt_D = optim.Adam(self.D.parameters(), lr=config.d_lr)

        # Loss function
        self.criterion = nn.BCELoss()

        # Dataset
        self.dataset = PointCloudDataset(config)
        self.data_loader = DataLoader(
            self.dataset,
            batch_size=min(config.batch_size, len(self.dataset)),
            shuffle=True,
            num_workers=0
        )

        # Fixed noise for generating samples
        self.fixed_z = torch.randn(4, config.latent_dim, device=self.device)

        # Training statistics
        self.d_losses = []
        self.g_losses = []

    def train(self):
        print(f"Starting training for {self.config.class_name} with {len(self.dataset)} samples")

        for epoch in range(self.config.epochs):
            epoch_d_losses = []
            epoch_g_losses = []

            for real in self.data_loader:
                real = real.to(self.device)
                batch_size = real.size(0)

                # Train discriminator
                self.opt_D.zero_grad()

                # Real samples
                real_labels = torch.ones(batch_size, 1, device=self.device)
                real_output = self.D(real)
                d_loss_real = self.criterion(real_output, real_labels)

                # Generated samples
                z = torch.randn(batch_size, self.config.latent_dim, device=self.device)
                fake = self.G(z)
                fake_labels = torch.zeros(batch_size, 1, device=self.device)
                fake_output = self.D(fake.detach())
                d_loss_fake = self.criterion(fake_output, fake_labels)

                # Total discriminator loss
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.opt_D.step()
                epoch_d_losses.append(d_loss.item())

                # Train generator
                self.opt_G.zero_grad()

                # Try to fool discriminator
                output = self.D(fake)
                g_loss = self.criterion(output, real_labels)
                g_loss.backward()
                self.opt_G.step()
                epoch_g_losses.append(g_loss.item())

            # Record average losses
            avg_d_loss = np.mean(epoch_d_losses)
            avg_g_loss = np.mean(epoch_g_losses)
            self.d_losses.append(avg_d_loss)
            self.g_losses.append(avg_g_loss)

            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.config.epochs} | D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}")

                # Save samples and models
                self.save_samples(epoch)
                self.save_model(epoch)

        # Save final model after training
        self.save_model(self.config.epochs - 1)
        self.plot_losses()
        print(f"Training completed for {self.config.class_name}")

    def save_samples(self, epoch):
        self.G.eval()
        with torch.no_grad():
            samples = self.G(self.fixed_z).cpu().numpy()

        # Save point cloud files
        for i, points in enumerate(samples):
            filename = os.path.join(self.config.sample_dir, f"epoch_{epoch}_sample_{i}.ply")
            self.save_point_cloud(points, filename)

        # Visualize samples
        self.visualize_samples(samples, epoch)

    def visualize_samples(self, samples, epoch):
        fig = plt.figure(figsize=(10, 10))
        fig.suptitle(f"{self.config.class_name} - Epoch {epoch + 1}", fontsize=16)

        for i in range(min(4, len(samples))):
            ax = fig.add_subplot(2, 2, i + 1, projection='3d')
            points = samples[i]

            # Use color mapping to enhance visualization
            colors = cm.viridis(np.linspace(0, 1, points.shape[0]))
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=5)

            # Set equal scale
            max_val = max(points.max(), abs(points.min())) + 0.1
            ax.set_xlim([-max_val, max_val])
            ax.set_ylim([-max_val, max_val])
            ax.set_zlim([-max_val, max_val])

            ax.set_axis_off()
            ax.set_title(f"Sample {i + 1}")

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.sample_dir, f"epoch_{epoch}_samples.png"), dpi=100)
        plt.close()

    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.d_losses, label="Discriminator Loss")
        plt.plot(self.g_losses, label="Generator Loss")
        plt.title(f"Training Loss for {self.config.class_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(self.config.sample_dir, "training_loss.png"), dpi=100)
        plt.close()

    def save_point_cloud(self, points, filename):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(filename, pcd)

    def save_model(self, epoch):
        torch.save(self.G.state_dict(),
                   os.path.join(self.config.sample_dir, f"G_{self.config.class_name}_epoch_{epoch}.pth"))
        torch.save(self.D.state_dict(),
                   os.path.join(self.config.sample_dir, f"D_{self.config.class_name}_epoch_{epoch}.pth"))


# Train GAN for a single class
def train_single_class(class_name, min_samples=3):
    print(f"\n{'=' * 50}")
    print(f"Training GAN for class: {class_name}")
    print(f"{'=' * 50}")

    try:
        # Check class directory
        class_dir = os.path.join("./data/shape_prior_dataset/point_cloud", class_name)
        if not os.path.exists(class_dir):
            print(f"Directory not found: {class_dir}")
            return False

        # Check sample count
        ply_files = [f for f in os.listdir(class_dir) if f.endswith('.ply')]
        if len(ply_files) < min_samples:
            print(f"Only {len(ply_files)} samples found (minimum {min_samples} required)")
            return False

        # Create configuration
        config = Config(class_name)

        # Train GAN
        gan = PointCloudGANTrainer(config)
        gan.train()

        # Generate final samples
        print(f"\nGenerating final samples for {class_name}...")
        gan.save_samples(config.epochs - 1)

        return True
    except Exception as e:
        print(f"\nTraining failed for {class_name}: {str(e)}")
        return False

def generate_point_cloud(normal_gray_data):
    """
    Input: dict {view_name: {"normal": PIL.Image, "depth": np.ndarray, ...}}
    Output: dict {view_name: o3d.geometry.PointCloud}
    """
    point_clouds = {}
    for view_name, data in normal_gray_data.items():
        depth = data["depth"]           # H×W, 0-1
        normal = np.asarray(data["normal"])  # H×W×3, 0-255
        h, w = depth.shape

        # Camera intrinsics (consistent with training)
        fx = fy = max(h, w) * 1.1
        cx, cy = w / 2, h / 2

        # Depth back-projection to 3D
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

        # Filter invalid depth
        mask = depth.reshape(-1) > 1e-4
        points = points[mask]

        # Build point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Normals (optional)
        normals = normal.reshape(-1, 3)[mask] / 255.0 * 2 - 1
        pcd.normals = o3d.utility.Vector3dVector(normals)

        point_clouds[view_name] = pcd

    return point_clouds

# Main function
def main():
    parser = argparse.ArgumentParser(description='3D Point Cloud GAN Trainer')
    parser.add_argument('--class_name', type=str, default='all',
                        help='Class to train or "all" for all classes')
    parser.add_argument('--min_samples', type=int, default=3,
                        help='Minimum samples per class (default: 3)')
    args = parser.parse_args()

    # Determine classes to train
    if args.class_name.lower() == 'all':
        classes_to_train = ALL_CLASSES
    else:
        if args.class_name in ALL_CLASSES:
            classes_to_train = [args.class_name]
        else:
            print(f"Error: Invalid class name: {args.class_name}")
            print(f"Valid classes: {', '.join(ALL_CLASSES)}")
            return

    print(f"\nStarting training for {len(classes_to_train)} classes")
    print(f"Minimum samples per class: {args.min_samples}")

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Train all specified classes
    success_count = 0
    for i, class_name in enumerate(classes_to_train):
        print(f"\nProcessing class {i + 1}/{len(classes_to_train)}: {class_name}")
        success = train_single_class(class_name, args.min_samples)
        if success:
            success_count += 1
            print(f"Successfully trained {class_name}")
        else:
            print(f"Skipped or failed to train {class_name}")

    print(f"\nTraining completed! Successfully trained {success_count}/{len(classes_to_train)} classes")


if __name__ == "__main__":
    main()