#!/usr/bin/env python3
"""
Complete implementation of HSDE (Hierarchical Semantic Disentanglement Encoder)
Includes ViT image encoder, CLIP text encoder, and 3D attention fusion mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from transformers import ViTModel, ViTConfig, CLIPTextModel, CLIPConfig
from transformers.models.clip.configuration_clip import CLIPTextConfig

logger = logging.getLogger(__name__)


@dataclass
class HSDEConfig:
    """HSDE configuration class"""
    # ViT configuration
    vit_model_name: str = "google/vit-base-patch16-224"
    vit_hidden_size: int = 768
    vit_num_attention_heads: int = 12
    vit_num_hidden_layers: int = 12

    # CLIP configuration
    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_hidden_size: int = 512
    clip_max_position_embeddings: int = 77

    # 3D spatial configuration
    spatial_resolution: int = 32  # 32x32x32 3D grid
    latent_dim: int = 256

    # Attention configuration
    num_attention_heads: int = 8
    attention_dropout: float = 0.1

    # Loss function weights
    contrastive_weight: float = 1.0
    entropy_weight: float = 0.1
    smoothness_weight: float = 0.01
    temperature: float = 0.07


class PositionalEncoding3D(nn.Module):
    """3D positional encoding"""

    def __init__(self, spatial_resolution: int, latent_dim: int):
        super().__init__()
        self.spatial_resolution = spatial_resolution
        self.latent_dim = latent_dim

        # Create 3D position coordinates
        coords = torch.meshgrid(
            torch.linspace(-1, 1, spatial_resolution),
            torch.linspace(-1, 1, spatial_resolution),
            torch.linspace(-1, 1, spatial_resolution),
            indexing='ij'
        )
        self.register_buffer('position_coords', torch.stack(coords, dim=-1).reshape(-1, 3))

        # Position encoding network
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, latent_dim)
        )

    def forward(self) -> torch.Tensor:
        """Return positional encoding"""
        return self.pos_encoder(self.position_coords)


class MultiHeadCrossAttention3D(nn.Module):
    """3D multi-head cross-attention mechanism"""

    def __init__(self, config: HSDEConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.latent_dim // config.num_attention_heads
        self.scale = self.head_dim ** -0.5

        # Query, key, value projections
        self.q_proj = nn.Linear(config.latent_dim, config.latent_dim)
        self.k_proj = nn.Linear(config.latent_dim, config.latent_dim)
        self.v_proj = nn.Linear(config.latent_dim, config.latent_dim)
        self.out_proj = nn.Linear(config.latent_dim, config.latent_dim)

        self.dropout = nn.Dropout(config.attention_dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: [batch_size, num_queries, latent_dim]
            key: [batch_size, num_keys, latent_dim]
            value: [batch_size, num_keys, latent_dim]
            attention_mask: [batch_size, num_queries, num_keys]
        Returns:
            attended_features: [batch_size, num_queries, latent_dim]
        """
        batch_size, num_queries, _ = query.shape
        num_keys = key.shape[1]

        # Project to multi-head representation
        q = self.q_proj(query).view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, num_keys, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, num_keys, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply attention mask
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)

        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attended = torch.matmul(attention_weights, v)

        # Combine multi-head
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, num_queries, self.num_heads * self.head_dim
        )

        return self.out_proj(attended)


class ViTImageEncoder(nn.Module):
    """ViT image encoder"""

    def __init__(self, config: HSDEConfig):
        super().__init__()
        self.config = config

        # Load pre-trained ViT model
        vit_config = ViTConfig(
            hidden_size=config.vit_hidden_size,
            num_attention_heads=config.vit_num_attention_heads,
            num_hidden_layers=config.vit_num_hidden_layers
        )
        self.vit = ViTModel(vit_config)

        # Adapter layer
        self.adapter = nn.Linear(config.vit_hidden_size, config.latent_dim)

        # Freeze early layers of ViT, only train later layers
        self.freeze_early_layers()

    def freeze_early_layers(self, freeze_layers: int = 6):
        """Freeze early layers of ViT"""
        for i, layer in enumerate(self.vit.encoder.layer):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [batch_size, 3, height, width]
        Returns:
            visual_features: [batch_size, num_patches, latent_dim]
        """
        # Extract ViT features
        vit_outputs = self.vit(images)
        hidden_states = vit_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # Adapt to target dimension
        visual_features = self.adapter(hidden_states)

        return visual_features


class CLIPTextEncoder(nn.Module):
    """CLIP text encoder"""

    def __init__(self, config: HSDEConfig):
        super().__init__()
        self.config = config

        # Create CLIP text configuration
        clip_text_config = CLIPTextConfig(
            hidden_size=config.clip_hidden_size,
            max_position_embeddings=config.clip_max_position_embeddings
        )

        # Build text encoder
        self.text_encoder = CLIPTextModel(clip_text_config)

        # Adapter layer
        self.adapter = nn.Linear(config.clip_hidden_size, config.latent_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        Returns:
            text_features: [batch_size, seq_len, latent_dim]
        """
        # Extract text features
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = text_outputs.last_hidden_state

        # Adapt to target dimension
        text_features = self.adapter(hidden_states)

        return text_features


class AttentionFusionModule(nn.Module):
    """3D attention fusion module"""

    def __init__(self, config: HSDEConfig):
        super().__init__()
        self.config = config

        # 3D positional encoding
        self.positional_encoding = PositionalEncoding3D(
            config.spatial_resolution, config.latent_dim
        )

        # 3D anchor queries
        self.anchor_queries = nn.Parameter(
            torch.randn(config.spatial_resolution ** 3, config.latent_dim)
        )

        # Multi-layer cross attention
        self.cross_attention_layers = nn.ModuleList([
            MultiHeadCrossAttention3D(config)
            for _ in range(4)
        ])

        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.latent_dim, config.latent_dim * 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(config.latent_dim * 4, config.latent_dim)
            )
            for _ in range(4)
        ])

        self.layer_norms_1 = nn.ModuleList([nn.LayerNorm(config.latent_dim) for _ in range(4)])
        self.layer_norms_2 = nn.ModuleList([nn.LayerNorm(config.latent_dim) for _ in range(4)])

    def forward(self, visual_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features: [batch_size, num_patches, latent_dim]
            text_features: [batch_size, seq_len, latent_dim]
        Returns:
            fused_features: [batch_size, num_anchors, latent_dim]
        """
        batch_size = visual_features.shape[0]

        # Expand anchor queries to batch size
        anchor_queries = self.anchor_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Add positional encoding
        pos_encoding = self.positional_encoding().unsqueeze(0).expand(batch_size, -1, -1)
        anchor_queries = anchor_queries + pos_encoding

        # Fuse features
        fused_features = anchor_queries

        for i, (attention_layer, ffn_layer, ln1, ln2) in enumerate(
                zip(self.cross_attention_layers, self.ffn_layers, self.layer_norms_1, self.layer_norms_2)
        ):
            # First layer: visual features as key and value
            if i % 2 == 0:
                attended = attention_layer(fused_features, visual_features, visual_features)
            # Second layer: text features as key and value
            else:
                attended = attention_layer(fused_features, text_features, text_features)

            # Residual connection and layer normalization
            fused_features = ln1(fused_features + attended)

            # Feed-forward network
            ffn_output = ffn_layer(fused_features)
            fused_features = ln2(fused_features + ffn_output)

        return fused_features


class SemanticVolumeDecoder(nn.Module):
    """Semantic volume decoder"""

    def __init__(self, config: HSDEConfig):
        super().__init__()
        self.config = config

        # Semantic classification head
        self.semantic_classifier = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim // 2),
            nn.ReLU(),
            nn.Linear(config.latent_dim // 2, 128)  # 128 semantic categories
        )

        # Bounding box regression head
        self.bbox_regressor = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim // 2),
            nn.ReLU(),
            nn.Linear(config.latent_dim // 2, 6)  # [x_min, y_min, z_min, x_max, y_max, z_max]
        )

        # Confidence prediction head
        self.confidence_predictor = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim // 2),
            nn.ReLU(),
            nn.Linear(config.latent_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, fused_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            fused_features: [batch_size, num_anchors, latent_dim]
        Returns:
            predictions: {
                'semantic_logits': [batch_size, num_anchors, 128],
                'bboxes': [batch_size, num_anchors, 6],
                'confidences': [batch_size, num_anchors, 1]
            }
        """
        semantic_logits = self.semantic_classifier(fused_features)
        bboxes = self.bbox_regressor(fused_features)
        confidences = self.confidence_predictor(fused_features)

        return {
            'semantic_logits': semantic_logits,
            'bboxes': bboxes,
            'confidences': confidences
        }


class HSDELoss(nn.Module):
    """HSDE loss function"""

    def __init__(self, config: HSDEConfig):
        super().__init__()
        self.config = config

        self.contrastive_loss = nn.CrossEntropyLoss()
        self.semantic_loss = nn.CrossEntropyLoss()
        self.bbox_loss = nn.SmoothL1Loss()

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: Model prediction results
            targets: Target labels
        Returns:
            losses: Various loss terms
        """
        losses = {}

        # Semantic classification loss
        if 'semantic_labels' in targets:
            semantic_loss = self.semantic_loss(
                predictions['semantic_logits'].reshape(-1, 128),
                targets['semantic_labels'].reshape(-1)
            )
            losses['semantic_loss'] = semantic_loss

        # Bounding box regression loss
        if 'bbox_targets' in targets:
            valid_mask = targets.get('bbox_valid_mask', torch.ones_like(targets['bbox_targets'][:, :, 0]))
            if valid_mask.sum() > 0:
                bbox_loss = self.bbox_loss(
                    predictions['bboxes'][valid_mask],
                    targets['bbox_targets'][valid_mask]
                )
                losses['bbox_loss'] = bbox_loss

        # Confidence loss
        if 'confidence_targets' in targets:
            confidence_loss = F.binary_cross_entropy(
                predictions['confidences'].squeeze(-1),
                targets['confidence_targets']
            )
            losses['confidence_loss'] = confidence_loss

        # Contrastive learning loss (simulated)
        if 'contrastive_labels' in targets:
            contrastive_loss = self.contrastive_loss(
                predictions.get('contrastive_logits', predictions['semantic_logits']),
                targets['contrastive_labels']
            )
            losses['contrastive_loss'] = contrastive_loss

        # Entropy regularization
        if 'entropy_loss' in predictions:
            losses['entropy_loss'] = predictions['entropy_loss'] * self.config.entropy_weight

        # Smoothness regularization
        if 'smoothness_loss' in predictions:
            losses['smoothness_loss'] = predictions['smoothness_loss'] * self.config.smoothness_weight

        return losses


class HSDE(nn.Module):
    """Complete HSDE model"""

    def __init__(self, config: Optional[HSDEConfig] = None):
        super().__init__()
        self.config = config or HSDEConfig()

        # Component initialization
        self.image_encoder = ViTImageEncoder(self.config)
        self.text_encoder = CLIPTextEncoder(self.config)
        self.fusion_module = AttentionFusionModule(self.config)
        self.volume_decoder = SemanticVolumeDecoder(self.config)
        self.loss_function = HSDELoss(self.config)

        # 3D spatial anchors
        self.register_buffer(
            'spatial_anchors',
            self.create_spatial_anchors(self.config.spatial_resolution)
        )

    def create_spatial_anchors(self, resolution: int) -> torch.Tensor:
        """Create 3D spatial anchors"""
        coords = torch.meshgrid(
            torch.linspace(-1, 1, resolution),
            torch.linspace(-1, 1, resolution),
            torch.linspace(-1, 1, resolution),
            indexing='ij'
        )
        return torch.stack(coords, dim=-1).reshape(-1, 3)

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                targets: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        Args:
            images: [batch_size, 3, height, width]
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            targets: Target labels
        Returns:
            results: Dictionary containing prediction results and losses
        """
        batch_size = images.shape[0]

        # Encode image features
        visual_features = self.image_encoder(images)

        # Encode text features
        text_features = self.text_encoder(input_ids, attention_mask)

        # 3D attention fusion
        fused_features = self.fusion_module(visual_features, text_features)

        # Decode semantic volume
        predictions = self.volume_decoder(fused_features)

        results = {
            'predictions': predictions,
            'fused_features': fused_features,
            'visual_features': visual_features,
            'text_features': text_features
        }

        # Calculate losses
        if targets is not None:
            losses = self.loss_function(predictions, targets)
            total_loss = sum(losses.values())
            results['losses'] = losses
            results['total_loss'] = total_loss

        return results

    def extract_semantic_volumes(self, images: torch.Tensor, texts: List[str],
                                 tokenizer) -> Dict[str, torch.Tensor]:
        """Extract semantic volume representations"""
        # Prepare inputs
        device = next(self.parameters()).device
        images = images.to(device)

        # Encode text
        text_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = text_inputs['input_ids'].to(device)
        attention_mask = text_inputs.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Forward pass
        with torch.no_grad():
            results = self.forward(images, input_ids, attention_mask)

        predictions = results['predictions']

        # Process prediction results
        semantic_volumes = []
        bboxes = []
        confidences = []

        for i in range(len(texts)):
            # Get semantic volumes with high confidence
            conf_mask = predictions['confidences'][i].squeeze(-1) > 0.5

            if conf_mask.sum() > 0:
                semantic_volumes.append(predictions['semantic_logits'][i][conf_mask])
                bboxes.append(predictions['bboxes'][i][conf_mask])
                confidences.append(predictions['confidences'][i][conf_mask])
            else:
                # If no high confidence ones, take top few
                top_k = min(5, predictions['confidences'][i].shape[0])
                top_indices = predictions['confidences'][i].squeeze(-1).topk(top_k).indices

                semantic_volumes.append(predictions['semantic_logits'][i][top_indices])
                bboxes.append(predictions['bboxes'][i][top_indices])
                confidences.append(predictions['confidences'][i][top_indices])

        return {
            'semantic_volumes': semantic_volumes,
            'bboxes': bboxes,
            'confidences': confidences,
            'spatial_anchors': self.spatial_anchors
        }


# Example usage and test code
if __name__ == "__main__":
    # Test HSDE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create configuration
    config = HSDEConfig()

    # Create model
    hsde = HSDE(config).to(device)

    # Test inputs
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    input_ids = torch.randint(0, 1000, (batch_size, 77)).to(device)

    # Forward pass
    with torch.no_grad():
        results = hsde.forward(images, input_ids)

    print("HSDE test results:")
    print(f"Input image shape: {images.shape}")
    print(f"Input text IDs shape: {input_ids.shape}")
    print(f"Fused features shape: {results['fused_features'].shape}")
    print(f"Semantic logits shape: {results['predictions']['semantic_logits'].shape}")
    print(f"Bounding boxes shape: {results['predictions']['bboxes'].shape}")
    print(f"Confidence shape: {results['predictions']['confidences'].shape}")

    # Test semantic volume extraction
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    texts = ["a window on the wall", "a chair in the room"]
    semantic_results = hsde.extract_semantic_volumes(images[:len(texts)], texts, tokenizer)

    print("\nSemantic volume extraction results:")
    print(f"Number of spatial anchors: {semantic_results['spatial_anchors'].shape[0]}")
    print(f"Number of semantic volumes: {[len(vol) for vol in semantic_results['semantic_volumes']]}")
    print(f"Number of bounding boxes: {[len(bbox) for bbox in semantic_results['bboxes']]}")

    # Calculate parameter count
    total_params = sum(p.numel() for p in hsde.parameters())
    trainable_params = sum(p.numel() for p in hsde.parameters() if p.requires_grad)

    print(f"\nModel parameter statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Parameter efficiency: {trainable_params / total_params * 100:.1f}%")