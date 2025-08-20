# multiview_gan.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import cv2
import random
import time
import glob
import re
import math
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


# Improved Generator Architecture - UNet Structure Residual Blocks
class ViewGenerator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, num_views=6):
        super(ViewGenerator, self).__init__()
        self.num_views = num_views

        # -------- encoder --------
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2)
        )
        self.enc2 = ResidualBlock(64, 128, stride=2)  # 128->64
        self.enc3 = ResidualBlock(128, 256, stride=2)  # 64->32
        self.enc4 = ResidualBlock(256, 512, stride=2)  # 32->16
        self.enc5 = ResidualBlock(512, 512, stride=1)  # 16->16

        # Perspective shift
        self.view_transform = nn.Sequential(
            nn.Conv2d(512 + num_views, 512, 3, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # -------- decoder --------
        # Input channel = skip the previous decoding layer output
        self.dec1 = ResidualBlock(512 + 512, 512)  # 16×16  1024->512
        self.dec2 = ResidualBlock(512 + 512, 256)  # 32×32  1024->256
        self.dec3 = ResidualBlock(256 + 256, 128)  # 64×64  512 ->128
        self.dec4 = ResidualBlock(128 + 128, 64)  # 128×128 256->64

        # Fix the final output layer
        self.final = nn.Sequential(
            nn.Conv2d(64, output_channels, 3, padding=1),  # 保持256x256尺寸
            nn.Tanh()
        )

    def forward(self, x, target_view):
        # ---- encode ----
        e1 = self.enc1(x)  # 256×256 -> 128×128   [B,64]
        e2 = self.enc2(e1)  # 128×128 -> 64×64     [B,128]
        e3 = self.enc3(e2)  # 64×64   -> 32×32     [B,256]
        e4 = self.enc4(e3)  # 32×32   -> 16×16     [B,512]
        e5 = self.enc5(e4)  # 16×16   -> 16×16     [B,512]

        # ---- Viewing Conditions ----
        view = target_view.view(target_view.size(0), -1, 1, 1)
        view = view.expand(-1, -1, e5.size(2), e5.size(3))
        x = torch.cat([e5, view], dim=1)
        x = self.view_transform(x)  # [B,512,16,16]

        # ---- decode ----
        # 16×16 -> 16×16
        d1 = self.dec1(torch.cat([x, e5], dim=1))  # [B,1024,16,16]
        d1_up = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=True)  # 32×32

        # Upsample encoder features to match decoder feature sizes
        e4_up = F.interpolate(e4, scale_factor=2, mode='bilinear', align_corners=True)  # 16->32
        d2 = self.dec2(torch.cat([d1_up, e4_up], dim=1))  # [B,1024,32,32]
        d2_up = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True)  # 64×64

        e3_up = F.interpolate(e3, scale_factor=2, mode='bilinear', align_corners=True)  # 32->64
        d3 = self.dec3(torch.cat([d2_up, e3_up], dim=1))  # [B,512,64,64]
        d3_up = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True)  # 128×128

        e2_up = F.interpolate(e2, scale_factor=2, mode='bilinear', align_corners=True)  # 64->128
        d4 = self.dec4(torch.cat([d3_up, e2_up], dim=1))  # [B,256,128,128]
        d4_up = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True)  # 256×256

        out = self.final(d4_up)  # [B,3,256,256]
        return out


# Residual blocks
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.in1 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm2d(out_channels)

        # Downsampling
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.InstanceNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.in2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


# Improved Discriminator Architecture - PatchGAN
class ViewDiscriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(ViewDiscriminator, self).__init__()

        # 70x70 PatchGAN
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # After outputting the feature map, add global average pooling
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.model(x)


# Multi-perspective datasets
class TankMultiViewDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_size=256, num_views=6):
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size
        self.num_views = num_views

        # Define perspective mapping
        self.view_mapping = {
            "front": 0,
            "back": 1,
            "side_one": 2,
            "side_two": 3,
            "top": 4,
            "bottom": 5
        }

        # Load all images
        self.all_images = glob.glob(os.path.join(root_dir, '*.png'))
        print(f"Found {len(self.all_images)} images in {root_dir}")

        # Group images by ID
        self.tank_groups = {}
        for img_path in self.all_images:
            tank_id, view_type = self._extract_id_and_view(os.path.basename(img_path))
            if tank_id and view_type:
                if tank_id not in self.tank_groups:
                    self.tank_groups[tank_id] = {}
                self.tank_groups[tank_id][view_type] = img_path

        # Only the model with the full view is kept
        self.valid_tanks = []
        for tank_id, views in self.tank_groups.items():
            if len(views) >= num_views:
                self.valid_tanks.append(tank_id)
                print(f"Added tank {tank_id} with {len(views)} views")

        print(f"Found {len(self.valid_tanks)} tanks with at least {num_views} views")

        if not self.valid_tanks:
            print("Warning: No tanks with complete views found!")

    def _extract_id_and_view(self, filename):
        """Extract the ID and viewing type from the file name"""
        # Supported file name formats: 1_front.png, 1_back.png, 1_side_one.png, 1_side_two.png, 1_top.png, 1_bottom.png
        match = re.search(r'^(\d+)_(front|back|side_one|side_two|top|bottom)\.png$', filename)
        if match:
            return match.group(1), match.group(2)
        return None, None

    def __len__(self):
        return len(self.valid_tanks) * 100

    def __getitem__(self, idx):

        tank_idx = idx % len(self.valid_tanks)
        tank_id = self.valid_tanks[tank_idx]
        views = self.tank_groups[tank_id]


        available_views = list(views.keys())


        source_view_type, target_view_type = random.sample(available_views, 2)
        source_img_path = views[source_view_type]
        target_img_path = views[target_view_type]


        source_img = Image.open(source_img_path).convert('RGB')
        target_img = Image.open(target_img_path).convert('RGB')


        if self.transform:
            source_img = self.transform(source_img)
            target_img = self.transform(target_img)

        # Create a viewing vector (unique thermal encoding)
        view_vector = torch.zeros(self.num_views)
        view_vector[self.view_mapping[target_view_type]] = 1.0

        return source_img, target_img, view_vector


# Perceived Loss - Use the VGG feature
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        weights = VGG16_Weights.DEFAULT
        vgg = vgg16(weights=weights).features[:15]  # 取前 15 层
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.feature_extractor = vgg

    def forward(self, generated, target):
        # Map [-1,1] to [0,1] and then do ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(generated.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(generated.device)

        gen_norm = (generated + 1) / 2
        gen_norm = (gen_norm - mean) / std
        target_norm = (target + 1) / 2
        target_norm = (target_norm - mean) / std

        gen_feat = self.feature_extractor(gen_norm)
        target_feat = self.feature_extractor(target_norm)
        return F.l1_loss(gen_feat, target_feat)


# Gradient Penalty (WGAN-GP)
def compute_gradient_penalty(D, real_samples, fake_samples):
    # Ensure that the real sample and the generated sample are the same size
    if real_samples.size() != fake_samples.size():
        fake_samples = F.interpolate(fake_samples, size=real_samples.shape[2:], mode='bilinear', align_corners=False)

    # Create random interpolation
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(real_samples.device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    # Compute the discriminator output
    d_interpolates = D(interpolates)

    # Create pseudo-labels for gradient calculations
    fake = torch.ones(d_interpolates.shape).to(real_samples.device)

    # Calculate the gradient
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_multiview_gan(dataset_path, epochs=20, batch_size=16, lr=0.0001, device='cuda'):

    timestamp = int(time.time())
    output_dir = f"./output/multiview_models/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)


    with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {lr}\n")
        f.write(f"Device: {device}\n")

    print(f"Output directory: {output_dir}")

    # Enhanced data transformation
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    dataset = TankMultiViewDataset(dataset_path, transform=transform, target_size=256)

    if len(dataset) == 0:
        print("Error: No valid tanks found for training. Exiting.")
        return None, None

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Initialize the model - discriminator structure
    generator = ViewGenerator(num_views=6).to(device)


    class SimplifiedDiscriminator(nn.Module):
        def __init__(self, input_channels=3):
            super(SimplifiedDiscriminator, self).__init__()
            self.model = nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )

        def forward(self, x):
            return self.model(x)

    discriminator = SimplifiedDiscriminator().to(device)

    # Initialize perceived loss
    perceptual_loss = PerceptualLoss().to(device)

    # Optimizer - Use the Adam optimizer, which is more suitable for GAN training
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Learning rate scheduler - uses cosine annealing
    scheduler_g = optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=epochs)
    scheduler_d = optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=epochs)

    print(f"Starting training on {device} with {len(dataset)} samples")
    print(f"Batch size: {batch_size}, Epochs: {epochs}")

    # Training statistics
    history = {
        'd_loss': [],
        'g_loss': [],
        'g_loss_adv': [],
        'g_loss_rec': [],
        'g_loss_perceptual': []
    }

    # Training Loop
    for epoch in range(epochs):
        d_loss_epoch = 0.0
        g_loss_epoch = 0.0
        g_loss_adv_epoch = 0.0
        g_loss_rec_epoch = 0.0
        g_loss_perceptual_epoch = 0.0

        if epoch != 0:
            print(f"Pausing for 20 seconds after epoch {epoch + 1} to cool down GPU...")
            time.sleep(20)

        for i, (source_imgs, target_imgs, view_vectors) in enumerate(dataloader):
            batch_size = source_imgs.size(0)


            source_imgs = source_imgs.to(device)
            target_imgs = target_imgs.to(device)
            view_vectors = view_vectors.to(device)

            # ---------------------
            #  Training discriminator
            # ---------------------
            optimizer_d.zero_grad()


            with torch.no_grad():
                generated_imgs = generator(source_imgs, view_vectors)

            # Loss of real images
            real_output = discriminator(target_imgs)
            d_loss_real = -torch.mean(real_output)

            # Loss of generated images
            fake_output = discriminator(generated_imgs.detach())
            d_loss_fake = torch.mean(fake_output)

            # Gradient Penalty - Adjustment factor of 10.0
            gradient_penalty = compute_gradient_penalty(
                discriminator, target_imgs.data, generated_imgs.data
            )

            # Total Discriminator Loss - Adjusts the gradient penalty factor
            d_loss = d_loss_real + d_loss_fake + 10.0 * gradient_penalty
            d_loss.backward()
            optimizer_d.step()

            # ---------------------
            #  Training Generator
            # ---------------------
            for _ in range(200):
                optimizer_g.zero_grad()

                # Regenerate the image
                generated_imgs = generator(source_imgs, view_vectors)

                # Fight losses
                fake_output = discriminator(generated_imgs)
                g_loss_adv = -torch.mean(fake_output)

                # Rebuild loss (L1 loss retains more details) - Reduce weight
                g_loss_rec = F.l1_loss(generated_imgs, target_imgs)

                # Perceived Loss - Adjust the weight
                g_loss_perceptual = perceptual_loss(generated_imgs, target_imgs)

                # Total Generator Loss - Adjust the weight
                # Original weight: g_loss_adv + 50.0 * g_loss_rec + 5.0 * g_loss_perceptual
                # New weights: Increased the weight of adversarial losses and decreased the weight of reconstruction losses
                g_loss = 5.0 * g_loss_adv + 20.0 * g_loss_rec + 2.0 * g_loss_perceptual
                g_loss.backward()
                optimizer_g.step()

                # Updated statistics
                g_loss_epoch += g_loss.item()
                g_loss_adv_epoch += g_loss_adv.item()
                g_loss_rec_epoch += g_loss_rec.item()
                g_loss_perceptual_epoch += g_loss_perceptual.item()

            # Updated discriminator loss statistics
            d_loss_epoch += d_loss.item()

            # Print progress
            if i % 10 == 0:
                print(f"[Epoch {epoch + 1}/{epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}] "
                      f"[Adv: {g_loss_adv.item():.4f}, Rec: {g_loss_rec.item():.4f}, Percep: {g_loss_perceptual.item():.4f}]")

        # Update learning rate
        scheduler_g.step()
        scheduler_d.step()

        # Calculate the average loss
        d_loss_avg = d_loss_epoch / len(dataloader)
        g_loss_avg = g_loss_epoch / (2 * len(dataloader))  # 注意：生成器训练了2倍次数
        g_loss_adv_avg = g_loss_adv_epoch / (2 * len(dataloader))
        g_loss_rec_avg = g_loss_rec_epoch / (2 * len(dataloader))
        g_loss_perceptual_avg = g_loss_perceptual_epoch / (2 * len(dataloader))

        # Save history
        history['d_loss'].append(d_loss_avg)
        history['g_loss'].append(g_loss_avg)
        history['g_loss_adv'].append(g_loss_adv_avg)
        history['g_loss_rec'].append(g_loss_rec_avg)
        history['g_loss_perceptual'].append(g_loss_perceptual_avg)

        print(f"Epoch {epoch + 1}/{epochs} summary: "
              f"D loss: {d_loss_avg:.4f}, G loss: {g_loss_avg:.4f} "
              f"[Adv: {g_loss_adv_avg:.4f}, Rec: {g_loss_rec_avg:.4f}, Percep: {g_loss_perceptual_avg:.4f}]")

        # Save the model checkpoint
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            torch.save(generator.state_dict(), os.path.join(output_dir, f"generator_epoch_{epoch + 1}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(output_dir, f"discriminator_epoch_{epoch + 1}.pth"))
            print(f"Saved checkpoint at epoch {epoch + 1}")

            # Save training history
            torch.save(history, os.path.join(output_dir, f"training_history_epoch_{epoch + 1}.pt"))

            # Generate sample images
            with torch.no_grad():
                sample_idx = random.randint(0, len(dataset) - 1)
                sample_source, sample_target, sample_view = dataset[sample_idx]
                sample_source = sample_source.unsqueeze(0).to(device)
                sample_view = sample_view.unsqueeze(0).to(device)

                generated = generator(sample_source, sample_view)
                generated = generated.squeeze(0).cpu()


                save_image(generated, os.path.join(output_dir, f"sample_epoch_{epoch + 1}.png"))


    torch.save(generator.state_dict(), os.path.join(output_dir, "multiview_generator_final.pth"))
    torch.save(discriminator.state_dict(), os.path.join(output_dir, "multiview_discriminator_final.pth"))
    torch.save(history, os.path.join(output_dir, "training_history_final.pt"))

    print("Training completed! Final model saved.")
    return generator, history


# Auxiliary function: save the image
def save_image(tensor, filename):
    tensor = tensor.clone().detach()
    tensor = tensor * 0.5 + 0.5
    tensor = tensor.clamp(0, 1)
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)


# Multi-view generation functions
def generate_multiview(input_image, angles=["front", "back", "side_one", "side_two", "top", "bottom"], model_path=None):
    """Multi-view images are generated using trained GAN models"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    generator = ViewGenerator(num_views=6).to(device)

    if model_path and os.path.exists(model_path):
        generator.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded pre-trained multiview generator")
    else:

        models_dir = "output/multiview_models"
        if os.path.exists(models_dir):
            model_dirs = sorted([d for d in os.listdir(models_dir)
                                 if os.path.isdir(os.path.join(models_dir, d))], reverse=True)
            if model_dirs:
                for model_dir in model_dirs:
                    latest_dir = os.path.join(models_dir, model_dir)
                    model_path = os.path.join(latest_dir, "multiview_generator_final.pth")
                    if os.path.exists(model_path):
                        generator.load_state_dict(torch.load(model_path, map_location=device))
                        print(f"Loaded model from {model_path}")
                        break
                else:
                    print("Warning: No generator model found. Using untrained model.")
            else:
                print("Warning: No model directories found. Using untrained model.")

    generator.eval()

    # Define perspective vector mapping
    view_mapping = {
        "front": [1, 0, 0, 0, 0, 0],
        "back": [0, 1, 0, 0, 0, 0],
        "side_one": [0, 0, 1, 0, 0, 0],
        "side_two": [0, 0, 0, 1, 0, 0],
        "top": [0, 0, 0, 0, 1, 0],
        "bottom": [0, 0, 0, 0, 0, 1]
    }

    # Prepare to enter the image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    input_img = Image.open(input_image).convert('RGB')
    original_size = input_img.size
    input_tensor = transform(input_img).unsqueeze(0).to(device)

    results = {}

    for angle in angles:
        # Get the perspective vector
        if angle in view_mapping:
            view_vector = torch.tensor(view_mapping[angle], dtype=torch.float32).unsqueeze(0).to(device)
        else:
            print(f"Warning: Unknown view angle '{angle}'. Using front view instead.")
            view_vector = torch.tensor(view_mapping["front"], dtype=torch.float32).unsqueeze(0).to(device)


        with torch.no_grad():
            generated = generator(input_tensor, view_vector)


        generated = generated.squeeze(0).cpu().detach()
        generated = (generated * 0.5) + 0.5  # 反归一化
        generated = transforms.ToPILImage()(generated)


        if generated.size != original_size:
            generated = generated.resize(original_size, Image.LANCZOS)

        results[angle] = generated
        print(f"Generated {angle} view")

    return results


if __name__ == "__main__":

    dataset_path = "./multiview_dataset/cifar/train"
    generator, history = train_multiview_gan(
        dataset_path,
        epochs=200,
        batch_size=32,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    if generator is None:
        print("Training failed. Cannot generate views.")
        exit(1)


    test_image = "./test_image/1.png"
    views = generate_multiview(test_image)


    output_dir = "generated_views"
    os.makedirs(output_dir, exist_ok=True)

    for angle, img in views.items():
        img_path = os.path.join(output_dir, f"{angle}_view.png")
        img.save(img_path)
        print(f"Saved {angle} view to {img_path}")