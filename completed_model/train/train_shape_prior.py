import time

import torch
import torch.nn as nn
import torch.optim as optim
import trimesh
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt


class PointNetEncoder(nn.Module):
    """Point cloud encoder"""

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
    """Shape prior decoder"""

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
    """Complete shape prior model"""

    def __init__(self, latent_dim=128, num_points=2048):
        super(ShapePriorModel, self).__init__()
        self.encoder = PointNetEncoder(latent_dim)
        self.decoder = ShapePriorDecoder(latent_dim, num_points)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


class ShapePriorDataset(Dataset):
    def __init__(self, root_dir, num_points=2048):
        self.root_dir = root_dir
        self.num_points = num_points
        self.classes = []
        self.samples = []
        self.invalid_files = []

        # Collect valid samples
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            self.classes.append(class_name)
            class_idx = len(self.classes) - 1

            for obj_file in os.listdir(class_dir):
                if not obj_file.endswith('.obj') or obj_file.startswith('original_'):
                    continue

                obj_path = os.path.join(class_dir, obj_file)
                self.samples.append((obj_path, class_idx))

        print(f"Loaded {len(self.samples)} models from {len(self.classes)} classes")

        # Save class list
        class_list_path = os.path.join(root_dir, "class_list.txt")
        with open(class_list_path, 'w') as f:
            for class_name in self.classes:
                f.write(f"{class_name}\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obj_path, class_idx = self.samples[idx]

        try:
            # Use trimesh to load mesh
            mesh = trimesh.load(obj_path)

            # Validate mesh
            if mesh.is_empty or len(mesh.faces) < 4:
                raise ValueError("Invalid mesh: too few faces")

            # Convert to Open3D format
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(np.array(mesh.vertices))
            o3d_mesh.triangles = o3d.utility.Vector3iVector(np.array(mesh.faces))

            # Basic cleanup
            o3d_mesh.remove_duplicated_vertices()
            o3d_mesh.remove_degenerate_triangles()
            o3d_mesh.remove_unreferenced_vertices()

            # Sample point cloud
            pcd = o3d_mesh.sample_points_uniformly(number_of_points=self.num_points)
            points = np.asarray(pcd.points)

            # Point cloud normalization (preserve scale)
            centroid = np.mean(points, axis=0)
            points -= centroid
            max_dist = np.max(np.linalg.norm(points, axis=1))
            if max_dist > 1e-6:
                points /= max_dist

        except Exception as e:
            if obj_path not in self.invalid_files:
                print(f"⚠️ Skipping invalid file {obj_path}: {str(e)}")
                self.invalid_files.append(obj_path)

            # Generate neutral replacement data (unit sphere)
            phi = np.random.uniform(0, np.pi, self.num_points)
            theta = np.random.uniform(0, 2 * np.pi, self.num_points)
            x = np.sin(phi) * np.cos(theta) * 0.5
            y = np.sin(phi) * np.sin(theta) * 0.5
            z = np.cos(phi) * 0.5
            points = np.vstack([x, y, z]).T

        return torch.tensor(points, dtype=torch.float32), class_idx


def train_shape_prior():
    # Configuration parameters
    root_dir = "../data/shape_prior_dataset/full"
    output_dir = "../output/shape_prior_models"
    os.makedirs(output_dir, exist_ok=True)

    num_points = 2048
    latent_dim = 128
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.001

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataset and data loader
    dataset = ShapePriorDataset(root_dir, num_points=num_points)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize model
    model = ShapePriorModel(latent_dim=latent_dim, num_points=num_points).to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Use LambdaLR scheduler instead of ReduceLROnPlateau
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 0.95 ** epoch
    )

    # Training statistics
    train_losses = []

    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        if epoch != 0:
            print(f"Pausing for 10 seconds after epoch {epoch + 1} to cool down GPU...")
            time.sleep(10)

        for points, _ in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            points = points.to(device)

            # Forward pass
            reconstructed, _ = model(points)

            # Calculate loss
            loss = criterion(reconstructed, points)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Update learning rate
        scheduler.step()

        # Calculate average loss
        epoch_loss = running_loss / len(dataloader)
        train_losses.append(epoch_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(output_dir, f"shape_prior_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'classes': dataset.classes
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    # Save final model
    final_model_path = os.path.join(output_dir, "shape_prior_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.title('Shape Prior Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    print("Training loss plot saved")

    print("Training completed!")


if __name__ == "__main__":
    train_shape_prior()