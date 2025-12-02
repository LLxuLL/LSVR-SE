#!/usr/bin/env python3
"""
Complete implementation of LC-NeRF (Language-Conditioned Neural Radiance Field)
Includes FiLM layer dynamic modulation, language condition response, and real-time material editing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from transformers import CLIPTextModel, CLIPTokenizer

logger = logging.getLogger(__name__)


@dataclass
class LCNerfConfig:
    """LC-NeRF configuration class"""
    # Positional encoding
    pos_encoding_dim: int = 10
    dir_encoding_dim: int = 4

    # Network structure
    hidden_dim: int = 256
    num_layers: int = 8

    # FiLM layer configuration
    film_dim: int = 256
    num_film_layers: int = 4

    # Language condition configuration
    text_embed_dim: int = 512

    # Rendering configuration
    num_samples_coarse: int = 64
    num_samples_fine: int = 128
    near: float = 0.1
    far: float = 10.0

    # Loss function weights
    photometric_weight: float = 1.0
    semantic_weight: float = 0.1
    view_consistency_weight: float = 0.01

    # Training configuration
    learning_rate: float = 5e-4
    warmup_steps: int = 1000


class PositionalEncoding(nn.Module):
    """Positional encoding"""

    def __init__(self, input_dim: int, num_frequencies: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.output_dim = input_dim * (2 * num_frequencies + 1)

        # Create frequencies
        frequencies = 2.0 ** torch.arange(num_frequencies, dtype=torch.float32)
        self.register_buffer('frequencies', frequencies)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., input_dim]
        Returns:
            encoded: [..., input_dim * (2 * num_frequencies + 1)]
        """
        # Original input
        encoded = [x]

        # Sine and cosine encoding
        for freq in self.frequencies:
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))

        return torch.cat(encoded, dim=-1)


class FiLMLayer(nn.Module):
    """FiLM (Feature-wise Linear Modulation) layer"""

    def __init__(self, feature_dim: int, condition_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.condition_dim = condition_dim

        # Generate scaling and shifting parameters
        self.scale_generator = nn.Sequential(
            nn.Linear(condition_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        self.shift_generator = nn.Sequential(
            nn.Linear(condition_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch_size, num_points, feature_dim]
            condition: [batch_size, condition_dim]
        Returns:
            modulated_features: [batch_size, num_points, feature_dim]
        """
        # Generate scaling and shifting
        scale = self.scale_generator(condition).unsqueeze(1)  # [batch_size, 1, feature_dim]
        shift = self.shift_generator(condition).unsqueeze(1)  # [batch_size, 1, feature_dim]

        # Apply FiLM modulation
        modulated_features = scale * features + shift

        return modulated_features


class DensityActivation(nn.Module):
    """Density activation function"""

    def __init__(self, threshold: float = 0.01):
        super().__init__()
        self.threshold = threshold

    def forward(self, raw_density: torch.Tensor) -> torch.Tensor:
        """
        Args:
            raw_density: [..., 1]
        Returns:
            density: [..., 1]
        """
        # Use softplus activation and add threshold control
        density = F.softplus(raw_density - 1.0)

        # Apply threshold gating
        gate = (density > self.threshold).float()
        density = density * gate

        return density


class ColorActivation(nn.Module):
    """Color activation function"""

    def forward(self, raw_color: torch.Tensor) -> torch.Tensor:
        """
        Args:
            raw_color: [..., 3]
        Returns:
            color: [..., 3]
        """
        # Use sigmoid activation and scale to [0,1]
        color = torch.sigmoid(raw_color)
        return color


class LanguageConditionedMLP(nn.Module):
    """Language-conditioned MLP"""

    def __init__(self, config: LCNerfConfig):
        super().__init__()
        self.config = config

        # Positional encoding
        self.pos_encoding = PositionalEncoding(3, config.pos_encoding_dim)
        self.dir_encoding = PositionalEncoding(3, config.dir_encoding_dim)

        pos_encoded_dim = 3 * (2 * config.pos_encoding_dim + 1)
        dir_encoded_dim = 3 * (2 * config.dir_encoding_dim + 1)

        # Density network
        self.density_layers = nn.ModuleList()
        self.density_layers.append(nn.Linear(pos_encoded_dim, config.hidden_dim))

        for i in range(config.num_layers - 1):
            if i < config.num_film_layers and i % 2 == 1:
                # Insert FiLM layer at specific layers
                self.density_layers.append(FiLMLayer(config.hidden_dim, config.text_embed_dim))
            else:
                self.density_layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))

        self.density_output = nn.Linear(config.hidden_dim, 1)
        self.density_activation = DensityActivation()

        # Skip connection
        self.skip_connection_idx = 4
        self.skip_linear = nn.Linear(config.hidden_dim + pos_encoded_dim, config.hidden_dim)

        # Color network
        self.color_layers = nn.ModuleList([
            nn.Linear(config.hidden_dim + dir_encoded_dim, config.hidden_dim // 2),
            nn.Linear(config.hidden_dim // 2, 3)
        ])
        self.color_activation = ColorActivation()

        # Feature extraction layer (for FiLM input)
        self.feature_extractor = nn.Linear(config.hidden_dim, config.text_embed_dim)

    def forward(self, positions: torch.Tensor, directions: torch.Tensor,
                text_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            positions: [batch_size, num_samples, 3]
            directions: [batch_size, num_samples, 3]
            text_embeddings: [batch_size, text_embed_dim]
        Returns:
            outputs: {
                'density': [batch_size, num_samples, 1],
                'color': [batch_size, num_samples, 3],
                'features': [batch_size, num_samples, hidden_dim]
            }
        """
        batch_size, num_samples, _ = positions.shape

        # Positional encoding
        pos_encoded = self.pos_encoding(positions)
        dir_encoded = self.dir_encoding(directions)

        # Density network forward pass
        h = pos_encoded

        for i, layer in enumerate(self.density_layers):
            if isinstance(layer, FiLMLayer):
                # FiLM layer needs text condition
                h = layer(h, text_embeddings)
            else:
                h = F.relu(layer(h))

            # Skip connection
            if i == self.skip_connection_idx:
                h = torch.cat([h, pos_encoded], dim=-1)
                h = F.relu(self.skip_linear(h))

        # Density output
        raw_density = self.density_output(h)
        density = self.density_activation(raw_density)

        # Feature extraction (for color network)
        features = h

        # Color network
        color_input = torch.cat([h, dir_encoded], dim=-1)
        color_h = F.relu(self.color_layers[0](color_input))
        raw_color = self.color_layers[1](color_h)
        color = self.color_activation(raw_color)

        return {
            'density': density,
            'color': color,
            'features': features
        }


class LanguageConditionedNeRF(nn.Module):
    """Language-conditioned NeRF model"""

    def __init__(self, config: Optional[LCNerfConfig] = None):
        super().__init__()
        self.config = config or LCNerfConfig()

        # Text encoder
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.text_projector = nn.Linear(512, self.config.text_embed_dim)

        # Language-conditioned MLP
        self.lc_mlp = LanguageConditionedMLP(self.config)

        # Freeze early layers of text encoder
        self.freeze_text_encoder_layers()

    def freeze_text_encoder_layers(self, freeze_layers: int = 6):
        """Freeze early layers of text encoder"""
        for i, layer in enumerate(self.text_encoder.text_model.encoder.layers):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False

    def encode_text(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode text"""
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use hidden state of [CLS] token
        text_embeddings = text_outputs.last_hidden_state[:, 0, :]  # [batch_size, 512]
        text_embeddings = self.text_projector(text_embeddings)  # [batch_size, text_embed_dim]
        return text_embeddings

    def sample_points(self, rays_o: torch.Tensor, rays_d: torch.Tensor,
                      near: float, far: float, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample points along rays"""
        batch_size, num_rays, _ = rays_o.shape

        # Calculate sampling depth
        t_vals = torch.linspace(0., 1., num_samples, device=rays_o.device)
        z_vals = near * (1. - t_vals) + far * t_vals
        z_vals = z_vals.unsqueeze(0).unsqueeze(0).expand(batch_size, num_rays, -1)

        # Calculate 3D point positions
        pts = rays_o.unsqueeze(2) + rays_d.unsqueeze(2) * z_vals.unsqueeze(3)

        # Calculate directions (normalized)
        dirs = rays_d.unsqueeze(2).expand(-1, -1, num_samples, -1)
        dirs = F.normalize(dirs, dim=-1)

        return pts, dirs

    def volume_rendering(self, density: torch.Tensor, color: torch.Tensor,
                         z_vals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Volume rendering"""
        batch_size, num_rays, num_samples, _ = density.shape

        # Calculate distance between adjacent points
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.tensor([1e10], device=density.device).expand(dists[..., :1].shape)], dim=-1)
        dists = dists * torch.norm(torch.ones_like(density[..., :1]), dim=-1)

        # Calculate alpha values
        alpha = 1.0 - torch.exp(-density.squeeze(-1) * dists)

        # Calculate weights
        weights = alpha * torch.cumprod(torch.cat([torch.ones((batch_size, num_rays, 1), device=alpha.device),
                                                   1. - alpha + 1e-10], dim=-1), dim=-1)[..., :-1]

        # Calculate final color
        rgb_map = torch.sum(weights.unsqueeze(-1) * color, dim=-2)

        # Calculate depth map
        depth_map = torch.sum(weights * z_vals, dim=-1)

        return rgb_map, depth_map

    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor,
                input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                near: Optional[float] = None, far: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            rays_o: [batch_size, num_rays, 3] Ray origins
            rays_d: [batch_size, num_rays, 3] Ray directions
            input_ids: [batch_size, seq_len] Text input IDs
            attention_mask: [batch_size, seq_len] Attention mask
            near: Near clipping plane
            far: Far clipping plane
        Returns:
            results: Dictionary containing rendering results and intermediate variables
        """
        batch_size, num_rays, _ = rays_o.shape
        near = near or self.config.near
        far = far or self.config.far

        # Encode text
        text_embeddings = self.encode_text(input_ids, attention_mask)

        # Sample points
        pts_coarse, dirs_coarse = self.sample_points(
            rays_o, rays_d, near, far, self.config.num_samples_coarse
        )

        # Coarse sampling rendering
        coarse_outputs = self.lc_mlp(pts_coarse, dirs_coarse, text_embeddings)

        # Calculate weights for importance sampling
        with torch.no_grad():
            z_vals_coarse = torch.linspace(near, far, self.config.num_samples_coarse,
                                           device=rays_o.device).unsqueeze(0).unsqueeze(0)
            weights = self.compute_weights(coarse_outputs['density'].squeeze(-1), z_vals_coarse)

            # Importance sampling
            z_vals_mid = 0.5 * (z_vals_coarse[..., 1:] + z_vals_coarse[..., :-1])
            z_samples = self.sample_pdf(z_vals_mid, weights[..., 1:-1], self.config.num_samples_fine)
            z_samples = z_samples.detach()

            # Merge coarse and fine sampling
            z_vals, _ = torch.sort(torch.cat([z_vals_coarse, z_samples], dim=-1), dim=-1)
            pts_fine = rays_o.unsqueeze(2) + rays_d.unsqueeze(2) * z_vals.unsqueeze(3)
            dirs_fine = F.normalize(rays_d.unsqueeze(2).expand(-1, -1, z_vals.shape[-1], -1), dim=-1)

        # Fine sampling rendering
        fine_outputs = self.lc_mlp(pts_fine, dirs_fine, text_embeddings)

        # Volume rendering
        rgb_map, depth_map = self.volume_rendering(fine_outputs['density'], fine_outputs['color'], z_vals)

        return {
            'rgb_map': rgb_map,
            'depth_map': depth_map,
            'z_vals': z_vals,
            'coarse_outputs': coarse_outputs,
            'fine_outputs': fine_outputs,
            'text_embeddings': text_embeddings
        }

    def compute_weights(self, density: torch.Tensor, z_vals: torch.Tensor) -> torch.Tensor:
        """Calculate weights for importance sampling"""
        batch_size, num_rays, num_samples = density.shape

        # Calculate distance between adjacent points
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.tensor([1e10], device=density.device).expand(dists[..., :1].shape)], dim=-1)

        # Calculate alpha values
        alpha = 1.0 - torch.exp(-density * dists)

        # Calculate weights
        weights = alpha * torch.cumprod(torch.cat([torch.ones((batch_size, num_rays, 1), device=alpha.device),
                                                   1. - alpha + 1e-10], dim=-1), dim=-1)[..., :-1]

        return weights

    def sample_pdf(self, bins: torch.Tensor, weights: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Sample from probability density function"""
        # Get cumulative distribution function
        pdf = F.relu(weights)
        pdf = pdf / (torch.sum(pdf, dim=-1, keepdim=True) + 1e-5)
        cdf = torch.cumsum(pdf[..., :-1], dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

        # Sample
        u = torch.rand(list(cdf.shape[:-1]) + [num_samples], device=cdf.device)

        # Find corresponding bin
        u = u.contiguous()
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds - 1), inds - 1)
        above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], dim=-1)

        # Handle boundary cases
        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(2).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(2).expand(matched_shape), 2, inds_g)

        # Interpolate
        denom = (cdf_g[..., 1] - cdf_g[..., 0])
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        return samples

    def render_edit(self, rays_o: torch.Tensor, rays_d: torch.Tensor,
                    text_embeddings: torch.Tensor,
                    edit_text_embeddings: Optional[torch.Tensor] = None,
                    edit_strength: float = 1.0) -> Dict[str, torch.Tensor]:
        """Render edited scene"""
        if edit_text_embeddings is not None:
            # Interpolate text embeddings
            interpolated_embeddings = (
                    (1 - edit_strength) * text_embeddings +
                    edit_strength * edit_text_embeddings
            )
        else:
            interpolated_embeddings = text_embeddings

        # Render using interpolated text embeddings
        return self.forward(rays_o, rays_d,
                            torch.zeros((rays_o.shape[0], 1), dtype=torch.long, device=rays_o.device),
                            text_embeddings=interpolated_embeddings)

    def extract_features(self, positions: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """Extract feature representation of 3D positions"""
        batch_size = positions.shape[0]
        num_positions = positions.shape[1]

        # Virtual direction
        directions = torch.ones_like(positions)
        directions = F.normalize(directions, dim=-1)

        # Get features
        outputs = self.lc_mlp(positions, directions, text_embeddings)

        return outputs['features']


class LCNerfLoss(nn.Module):
    """LC-NeRF loss function"""

    def __init__(self, config: LCNerfConfig):
        super().__init__()
        self.config = config

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: Model predictions
            targets: Target values
        Returns:
            losses: Loss dictionary
        """
        losses = {}

        # Photometric loss
        if 'rgb_gt' in targets:
            photometric_loss = F.mse_loss(predictions['rgb_map'], targets['rgb_gt'])
            losses['photometric_loss'] = photometric_loss * self.config.photometric_weight

        # Semantic contrastive loss
        if 'semantic_targets' in targets:
            # Could implement CLIP-style contrastive loss here
            semantic_loss = F.cosine_embedding_loss(
                predictions['text_embeddings'],
                targets['semantic_targets'],
                torch.ones(targets['semantic_targets'].shape[0], device=targets['semantic_targets'].device)
            )
            losses['semantic_loss'] = semantic_loss * self.config.semantic_weight

        # View consistency loss
        if 'multi_view_rgb' in targets:
            # Calculate consistency between different views
            view_consistency_loss = 0
            for i in range(len(targets['multi_view_rgb'])):
                for j in range(i + 1, len(targets['multi_view_rgb'])):
                    view_diff = F.mse_loss(
                        targets['multi_view_rgb'][i],
                        targets['multi_view_rgb'][j]
                    )
                    view_consistency_loss += view_diff

            if len(targets['multi_view_rgb']) > 1:
                view_consistency_loss /= (len(targets['multi_view_rgb']) * (len(targets['multi_view_rgb']) - 1) / 2)
                losses['view_consistency_loss'] = view_consistency_loss * self.config.view_consistency_weight

        # Total loss
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss

        return losses


class LCNerfRenderer:
    """LC-NeRF renderer"""

    def __init__(self, model: LanguageConditionedNeRF, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()

    def render_image(self, camera_params: Dict[str, torch.Tensor],
                     text_input: str,
                     tokenizer,
                     image_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """Render complete image"""
        height, width = image_size

        # Generate rays
        rays_o, rays_d = self.generate_rays(camera_params, height, width)

        # Encode text
        text_inputs = tokenizer(text_input, return_tensors="pt", padding=True, truncation=True)
        input_ids = text_inputs['input_ids'].to(self.device)
        attention_mask = text_inputs.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Chunk rendering to avoid memory overflow
        chunk_size = 1024
        num_rays = height * width

        rgb_maps = []

        with torch.no_grad():
            for i in range(0, num_rays, chunk_size):
                end_idx = min(i + chunk_size, num_rays)

                chunk_rays_o = rays_o[:, i:end_idx]
                chunk_rays_d = rays_d[:, i:end_idx]

                # Render chunk
                results = self.model(chunk_rays_o, chunk_rays_d, input_ids, attention_mask)
                rgb_maps.append(results['rgb_map'])

        # Merge results
        rgb_map = torch.cat(rgb_maps, dim=1)
        rgb_image = rgb_map.reshape(1, height, width, 3)

        # Convert to numpy
        rgb_image = rgb_image.squeeze(0).cpu().numpy()
        rgb_image = (rgb_image * 255).astype(np.uint8)

        return rgb_image

    def generate_rays(self, camera_params: Dict[str, torch.Tensor],
                      height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate camera rays"""
        # Implement camera ray generation logic
        # Simplified implementation here, should calculate based on camera parameters
        device = self.device

        # Create pixel grid
        i, j = torch.meshgrid(torch.linspace(0, width - 1, width, device=device),
                              torch.linspace(0, height - 1, height, device=device), indexing='ij')

        # Convert to camera coordinates
        # Need actual camera intrinsic and extrinsic parameters
        rays_o = torch.zeros(height, width, 3, device=device)
        rays_d = torch.stack([(i - width / 2) / width,
                              -(j - height / 2) / height,
                              -torch.ones_like(i)], dim=-1)

        rays_d = F.normalize(rays_d, dim=-1)

        return rays_o.unsqueeze(0), rays_d.unsqueeze(0)

    def render_edit(self, camera_params: Dict[str, torch.Tensor],
                    original_text: str,
                    edit_text: str,
                    tokenizer,
                    edit_strength: float = 1.0,
                    image_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """Render edited image"""
        height, width = image_size

        # Generate rays
        rays_o, rays_d = self.generate_rays(camera_params, height, width)

        # Encode original text and edit text
        orig_text_inputs = tokenizer(original_text, return_tensors="pt", padding=True, truncation=True)
        orig_input_ids = orig_text_inputs['input_ids'].to(self.device)
        orig_attention_mask = orig_text_inputs.get('attention_mask', None)
        if orig_attention_mask is not None:
            orig_attention_mask = orig_attention_mask.to(self.device)

        edit_text_inputs = tokenizer(edit_text, return_tensors="pt", padding=True, truncation=True)
        edit_input_ids = edit_text_inputs['input_ids'].to(self.device)
        edit_attention_mask = edit_text_inputs.get('attention_mask', None)
        if edit_attention_mask is not None:
            edit_attention_mask = edit_attention_mask.to(self.device)

        # Get text embeddings
        with torch.no_grad():
            orig_text_embeddings = self.model.encode_text(orig_input_ids, orig_attention_mask)
            edit_text_embeddings = self.model.encode_text(edit_input_ids, edit_attention_mask)

        # Chunk rendering
        chunk_size = 1024
        num_rays = height * width

        rgb_maps = []

        with torch.no_grad():
            for i in range(0, num_rays, chunk_size):
                end_idx = min(i + chunk_size, num_rays)

                chunk_rays_o = rays_o[:, i:end_idx]
                chunk_rays_d = rays_d[:, i:end_idx]

                # Render edited chunk
                results = self.model.render_edit(
                    chunk_rays_o, chunk_rays_d,
                    orig_text_embeddings, edit_text_embeddings,
                    edit_strength
                )
                rgb_maps.append(results['rgb_map'])

        # Merge results
        rgb_map = torch.cat(rgb_maps, dim=1)
        rgb_image = rgb_map.reshape(1, height, width, 3)

        # Convert to numpy
        rgb_image = rgb_image.squeeze(0).cpu().numpy()
        rgb_image = (rgb_image * 255).astype(np.uint8)

        return rgb_image


# Example usage and test code
if __name__ == "__main__":
    # Test LC-NeRF
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create configuration
    config = LCNerfConfig()

    # Create model
    lc_nerf = LanguageConditionedNeRF(config).to(device)

    # Test inputs
    batch_size = 2
    num_rays = 1024

    rays_o = torch.randn(batch_size, num_rays, 3).to(device)
    rays_d = F.normalize(torch.randn(batch_size, num_rays, 3), dim=-1).to(device)
    input_ids = torch.randint(0, 1000, (batch_size, 77)).to(device)

    # Forward pass
    with torch.no_grad():
        results = lc_nerf.forward(rays_o, rays_d, input_ids)

    print("LC-NeRF test results:")
    print(f"Ray origin shape: {rays_o.shape}")
    print(f"Ray direction shape: {rays_d.shape}")
    print(f"Input text IDs shape: {input_ids.shape}")
    print(f"Rendered image shape: {results['rgb_map'].shape}")
    print(f"Depth map shape: {results['depth_map'].shape}")

    # Test text encoding
    text_embeddings = lc_nerf.encode_text(input_ids)
    print(f"Text embedding shape: {text_embeddings.shape}")

    # Test feature extraction
    positions = torch.randn(batch_size, 100, 3).to(device)
    features = lc_nerf.extract_features(positions, text_embeddings)
    print(f"Extracted features shape: {features.shape}")

    # Calculate parameter count
    total_params = sum(p.numel() for p in lc_nerf.parameters())
    trainable_params = sum(p.numel() for p in lc_nerf.parameters() if p.requires_grad)

    print(f"\nModel parameter statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Parameter efficiency: {trainable_params / total_params * 100:.1f}%")

    # Test renderer
    renderer = LCNerfRenderer(lc_nerf, device)

    # Simulate camera parameters
    camera_params = {
        'intrinsics': torch.eye(3, device=device),
        'extrinsics': torch.eye(4, device=device)
    }

    print(f"\nRenderer initialization complete")
    print(f"Device: {device}")
    print(f"Model ready for rendering")