import cv2
import numpy as np
from PIL import Image
import os
import torch
from torchvision.transforms import ToTensor, transforms

# Add MiDaS depth estimation model
try:
    from torch.hub import load

    midas = load("intel-isl/MiDaS", "MiDaS_small").eval()
    print("Loaded MiDaS depth estimation model")
    midas_transform = transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
except:
    print("Failed to load MiDaS model, using fallback depth estimation")
    midas = None


def generate_normal_gray(images_dict):
    """Improved normal and depth map generation"""
    print("Using improved normal and depth generator")
    results = {}
    for angle, img in images_dict.items():
        print(f"Processing {angle} view. . . ")
        # Ensure image is in RGB format
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_cv = np.array(img)[:, :, :3].copy()

        # Generate grayscale image
        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

        # Improved depth estimation (using MiDaS model)
        depth = estimate_depth(img)

        # Create normal map (based on depth map)
        grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=5)

        # Create normal vector
        normal = np.dstack((-grad_x, -grad_y, np.ones_like(depth)))
        norm = np.linalg.norm(normal, axis=2)
        norm = np.maximum(norm, 1e-7)  # Avoid division by zero
        normal[:, :, 0] /= norm
        normal[:, :, 1] /= norm
        normal[:, :, 2] /= norm

        # Convert to RGB normal map (0-255)
        normal_rgb = (normal + 1.0) * 127.5
        normal_rgb = np.clip(normal_rgb, 0, 255).astype(np.uint8)

        results[angle] = {
            "normal": Image.fromarray(normal_rgb),
            "gray": Image.fromarray(gray),
            "depth": depth
        }
        print(f"Generated improved normal and depth for {angle} view")
    return results


def estimate_depth(image):
    """Estimate depth map using MiDaS model"""
    # If MiDaS is available, use it
    if midas is not None:
        # Convert image to model input format
        input_image = image.resize((384, 384))
        input_tensor = midas_transform(input_image).unsqueeze(0)

        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tensor = input_tensor.to(device)
        midas.to(device)

        # Predict depth
        with torch.no_grad():
            prediction = midas(input_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # Convert to numpy and normalize
        depth = prediction.cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth

    # Use improved fallback method when MiDaS is not available
    print("Using fallback depth estimation")
    # Convert to OpenCV format
    if isinstance(image, Image.Image):
        image = np.array(image)[:, :, :3]

    # Convert to HSV space
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    value = hsv[:, :, 2].astype(np.float32) / 255.0

    # Create edge map
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Distance transform
    distance_map = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)

    # Combine saturation and value
    combined = 0.7 * distance_map + 0.3 * saturation * value * 255

    # Normalize depth map
    depth = cv2.normalize(combined, None, 0, 1, cv2.NORM_MINMAX)
    return depth
