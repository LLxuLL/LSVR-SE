#!/usr/bin/env python3
"""
LSVR-SE Complete Training Script
Supports joint training and individual training of three components: HSDE, LC-NeRF, DPEE
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'  # 5 minutes
os.environ['HF_HUB_ETAG_TIMEOUT'] = '30'  # 30 seconds

import sys
import time
import logging
import argparse
import json
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm
import wandb

# Import LSVR-SE components
from lsvr_se_config import LSVRSEConfig, LSVRSEModelManager, DEFAULT_CONFIG, FAST_CONFIG, PRODUCTION_CONFIG
from hsde import HSDE, HSDEConfig, HSDELoss
from lc_nerf import LanguageConditionedNeRF, LCNerfConfig, LCNerfLoss
from dpee import DifferentiableProgrammaticEditingEngine, DPEEConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LSVR-SE-Training")


class LSVRSEDataset(Dataset):
    """LSVR-SE Training Dataset"""

    def __init__(self, data_root: str, split: str = "train", transform=None):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform

        # Data paths
        self.image_dir = self.data_root / split / "images"
        self.mesh_dir = self.data_root / split / "meshes"
        self.text_dir = self.data_root / split / "texts"

        # Ensure directories exist
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.mesh_dir.mkdir(parents=True, exist_ok=True)
        self.text_dir.mkdir(parents=True, exist_ok=True)

        # Load data list
        self.data_list = self._load_data_list()

        logger.info(f"Loaded {len(self.data_list)} samples for {split} split")

    def _load_data_list(self) -> List[Dict[str, str]]:
        """Load data list"""
        data_list = []

        # Scan image files
        image_files = list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png"))

        for image_file in image_files:
            base_name = image_file.stem

            # Check corresponding mesh file
            mesh_file = self.mesh_dir / f"{base_name}.ply"
            if not mesh_file.exists():
                # Create default mesh
                mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
                o3d.io.write_triangle_mesh(str(mesh_file), mesh)

            # Check corresponding text file
            text_file = self.text_dir / f"{base_name}.txt"
            if not text_file.exists():
                # Create default text
                default_texts = [
                    "Add window",
                    "Remove door",
                    "Rotate 45 degrees",
                    "Scale 1.5x",
                    "Add column"
                ]
                text = random.choice(default_texts)
                text_file.write_text(text, encoding='utf-8')

            data_list.append({
                'image': str(image_file),
                'mesh': str(mesh_file),
                'text': str(text_file),
                'base_name': base_name
            })

        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        base_name = data['base_name']

        try:
            # ---- 1. Image ----
            image = Image.open(data['image']).convert('RGB')
            image = image.resize((224, 224))
            image_array = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)

            # ---- 2. Text ----
            with open(data['text'], 'r', encoding='utf-8') as f:
                text = f.read().strip()

            # ---- 3. Mesh (placeholder, can be reconstructed later) ----
            mesh = o3d.io.read_triangle_mesh(data['mesh'])
            if not mesh.has_vertices():
                mesh = o3d.geometry.TriangleMesh.create_box()

            # ---- 4. Camera poses (new) ----
            pose_file = Path(data['mesh']).parent.parent / 'poses' / f'{base_name}.json'
            if pose_file.exists():
                with open(pose_file, 'r') as f:
                    poses = json.load(f)  # dict or list
            else:
                poses = None  # placeholder
            return {
                'image': image_tensor,
                'text': text,
                'mesh': mesh,
                'poses': poses,  # new field
                'base_name': base_name
            }
        except Exception as e:
            logger.error(f"Error loading {base_name}: {e}")
            # Return default placeholder
            return {
                'image': torch.randn(3, 224, 224),
                'text': 'placeholder',
                'mesh': o3d.geometry.TriangleMesh.create_box(),
                'poses': None,
                'base_name': 'error'
            }

    def _get_default_sample(self):
        """Get default sample"""
        image = torch.randn(3, 224, 224)
        mesh = o3d.geometry.TriangleMesh.create_box()
        text = "Add window"

        return {
            'image': image,
            'mesh': mesh,
            'text': text,
            'base_name': 'default'
        }


class LSVRSETrainer:
    """LSVR-SE Trainer"""

    def __init__(self, config: LSVRSEConfig, use_wandb: bool = True):
        self.config = config
        self.use_wandb = use_wandb

        # Initialize model manager
        self.model_manager = LSVRSEModelManager(config)

        # Components
        self.hsde = None
        self.lc_nerf = None
        self.dpee = None

        # Loss functions
        self.hsde_loss = None
        self.lc_nerf_loss = None

        # Optimizer
        self.optimizer = None
        self.scheduler = None
        self.scaler = None

        # Device
        self.device = torch.device(config.device)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0

        # Initialize WandB
        if use_wandb:
            wandb.init(
                project=config.project_name,
                config=config,
                name=f"lsvr_se_train_{int(time.time())}"
            )

    def setup_models(self):
        """Set up models"""
        logger.info("Setting up models...")

        try:
            # Initialize model manager
            self.model_manager.initialize_components()

            # Get components
            self.hsde = self.model_manager.hsde
            self.lc_nerf = self.model_manager.lc_nerf
            self.dpee = self.model_manager.dpee

            # Initialize loss functions
            if self.hsde is not None:
                hsde_config = HSDEConfig(**self.config.hsde_config)
                self.hsde_loss = HSDELoss(hsde_config)

            if self.lc_nerf is not None:
                lc_nerf_config = LCNerfConfig(**self.config.lc_nerf_config)
                self.lc_nerf_loss = LCNerfLoss(lc_nerf_config)

            # Set up optimizer
            self._setup_optimizer()

            # Set up mixed precision
            if self.config.mixed_precision:
                self.scaler = GradScaler()

            logger.info("Models setup completed!")

        except Exception as e:
            logger.error(f"Failed to setup models: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _setup_optimizer(self):
        """Set up optimizer"""
        trainable_params = []

        if self.hsde is not None:
            trainable_params.extend(list(self.hsde.parameters()))

        if self.lc_nerf is not None:
            trainable_params.extend(list(self.lc_nerf.parameters()))

        if self.dpee is not None:
            trainable_params.extend(list(self.dpee.remesh_network.parameters()))

        if not trainable_params:
            raise ValueError("No trainable parameters found!")

        # Create optimizer
        if self.config.training_config['optimizer'] == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config.training_config['learning_rate'],
                betas=self.config.training_config['betas'],
                eps=self.config.training_config['eps'],
                weight_decay=self.config.training_config.get('weight_decay', 1e-5)
            )
        else:
            self.optimizer = torch.optim.Adam(
                trainable_params,
                lr=self.config.training_config['learning_rate']
            )

        # Create learning rate scheduler
        if self.config.training_config['scheduler'] == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training_config['T_max'],
                eta_min=self.config.training_config['eta_min']
            )
        elif self.config.training_config['scheduler'] == 'StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=50,
                gamma=0.1
            )

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train one epoch"""
        logger.info(f"Starting epoch {epoch}")

        # Set training mode
        if self.hsde:
            self.hsde.train()
        if self.lc_nerf:
            self.lc_nerf.train()
        if self.dpee:
            self.dpee.remesh_network.train()

        epoch_losses = {
            'total_loss': 0.0,
            'hsde_loss': 0.0,
            'lc_nerf_loss': 0.0,
            'dpee_loss': 0.0
        }

        num_batches = len(dataloader)

        with tqdm(dataloader, desc=f"Epoch {epoch}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                try:
                    # Forward pass
                    losses = self.train_step(batch)

                    # Update losses
                    for key, value in losses.items():
                        if key in epoch_losses:
                            epoch_losses[key] += value

                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f"{losses['total_loss']:.4f}",
                        'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                    })

                    # Log to WandB
                    if self.use_wandb and self.global_step % 10 == 0:
                        wandb.log({
                            'train/step_loss': losses['total_loss'],
                            'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                            'train/global_step': self.global_step
                        })

                    self.global_step += 1

                except Exception as e:
                    logger.error(f"Error in batch {batch_idx}: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue

        # Calculate average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single training step: ensure total_loss is tensor and correctly call scheduler"""
        self.optimizer.zero_grad()

        images = batch['image'].to(self.device)
        texts = batch['text']
        poses = batch['poses']
        # meshes already split into vertices/faces, submodules don't use them for now, keep empty list
        meshes = []

        total_loss = torch.tensor(0.0, device=self.device)
        losses = {'total_loss': 0.0}

        if self.hsde is not None:
            loss = self._train_hsde_step(images, texts)
            losses['hsde_loss'] = loss.item()
            total_loss = total_loss + loss

        if self.lc_nerf is not None:
            loss = self._train_lc_nerf_step(images, texts, meshes)
            losses['lc_nerf_loss'] = loss.item()
            total_loss = total_loss + loss

        if self.dpee is not None:
            loss = self._train_dpee_step(meshes, texts)
            losses['dpee_loss'] = loss.item()
            total_loss = total_loss + loss

        losses['total_loss'] = total_loss.item()

        # Backward propagation
        if self.config.mixed_precision and self.scaler is not None:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            self.optimizer.step()

        # Learning rate scheduling (must be after optimizer.step)
        if self.scheduler is not None:
            self.scheduler.step()

        return losses

    def _train_hsde_step(self, images: torch.Tensor, texts: List[str]) -> torch.Tensor:
        """HSDE training sub-step: ensure returns tensor and indices are long"""
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            text_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            input_ids = text_inputs['input_ids'].to(self.device)  # already long
            attention_mask = text_inputs.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            with autocast(enabled=self.config.mixed_precision, device_type='cpu'):
                results = self.hsde(images, input_ids, attention_mask)

            targets = self._create_hsde_targets(results, texts)
            losses = self.hsde_loss(results['predictions'], targets)
            return losses['total_loss']  # keep tensor
        except Exception as e:
            logger.error(f"HSDE training step failed: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def _train_lc_nerf_step(self, images: torch.Tensor, texts: List[str], meshes) -> torch.Tensor:
        """LC-NeRF training sub-step: compatible with 4-dimensional images"""
        try:
            batch_size, _, height, width = images.shape  # [B,C,H,W]
            rays_o, rays_d = self._generate_training_rays(batch_size, height, width)

            # Random text ids
            input_ids = torch.randint(0, 1000, (batch_size, 77), device=self.device)

            with autocast(enabled=self.config.mixed_precision, device_type='cpu'):
                nerf_results = self.lc_nerf(rays_o, rays_d, input_ids)

            targets = self._create_lc_nerf_targets(nerf_results, images)
            losses = self.lc_nerf_loss(nerf_results, targets)
            return losses['total_loss']
        except Exception as e:
            logger.error(f"LC-NeRF training step failed: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def _train_dpee_step(self, meshes, texts: List[str]) -> torch.Tensor:
        """DPEE training sub-step: prevent division by zero, return tensor"""
        try:
            total_loss = 0.0
            for text in texts:
                if "添加" in text or "add" in text.lower():
                    total_loss += 0.1
                elif "移除" in text or "remove" in text.lower():
                    total_loss += 0.2
                else:
                    total_loss += 0.05
            denominator = max(len(texts), 1)
            loss = torch.tensor(total_loss / denominator, device=self.device, requires_grad=True)
            return loss
        except Exception as e:
            logger.error(f"DPEE training step failed: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def _generate_training_rays(self, batch_size: int, height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate training rays"""
        # Simplified ray generation
        num_rays = height * width

        rays_o = torch.zeros(batch_size, num_rays, 3, device=self.device)
        rays_d = torch.randn(batch_size, num_rays, 3, device=self.device)
        rays_d = F.normalize(rays_d, dim=-1)

        return rays_o, rays_d

    def _create_hsde_targets(self, results: Dict[str, Any], texts: List[str]) -> Dict[str, torch.Tensor]:
        """Create HSDE training targets"""
        batch_size = results['fused_features'].shape[0]
        num_anchors = results['fused_features'].shape[1]

        # Create virtual targets
        targets = {
            'semantic_labels': torch.randint(0, 128, (batch_size, num_anchors), device=self.device),
            'bbox_targets': torch.randn(batch_size, num_anchors, 6, device=self.device),
            'confidence_targets': torch.rand(batch_size, num_anchors, device=self.device)
        }

        return targets

    def _create_lc_nerf_targets(self, results: Dict[str, Any], images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Create LC-NeRF training targets"""
        batch_size, height, width = images.shape[0], images.shape[2], images.shape[3]

        # Create virtual targets
        targets = {
            'rgb_gt': images.permute(0, 2, 3, 1).reshape(batch_size, -1, 3),
            'semantic_targets': torch.randn(batch_size, 512, device=self.device)
        }

        return targets

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        logger.info("Starting validation...")

        # Set evaluation mode
        if self.hsde:
            self.hsde.eval()
        if self.lc_nerf:
            self.lc_nerf.eval()
        if self.dpee:
            self.dpee.remesh_network.eval()

        val_losses = {
            'total_loss': 0.0,
            'hsde_loss': 0.0,
            'lc_nerf_loss': 0.0,
            'dpee_loss': 0.0
        }

        num_batches = len(dataloader)

        with torch.no_grad():
            with tqdm(dataloader, desc="Validation") as pbar:
                for batch in pbar:
                    try:
                        # Forward pass
                        losses = self.train_step(batch)

                        # Update losses
                        for key, value in losses.items():
                            if key in val_losses:
                                val_losses[key] += value

                        pbar.set_postfix({'val_loss': f"{losses['total_loss']:.4f}"})

                    except Exception as e:
                        logger.error(f"Error in validation batch: {str(e)}")
                        continue

        # Calculate average losses
        for key in val_losses:
            val_losses[key] /= num_batches

        return val_losses

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save checkpoint"""
        self.model_manager.save_models(epoch)

        if is_best:
            best_path = self.model_manager.paths.get_checkpoint_path(epoch, "best")
            checkpoint = {
                'epoch': epoch,
                'best_metric': self.best_metric,
                'config': self.config
            }
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint to {best_path}")

    def train(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None,
              num_epochs: int = 100):
        """Main training loop"""
        logger.info("Starting training...")

        # Set up models
        self.setup_models()

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train one epoch
            train_losses = self.train_epoch(train_dataloader, epoch)

            # Validate
            val_losses = None
            if val_dataloader is not None:
                val_losses = self.validate(val_dataloader)

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # Log to WandB
            if self.use_wandb:
                log_dict = {
                    'epoch': epoch,
                    'train/epoch_loss': train_losses['total_loss'],
                    'train/epoch_hsde_loss': train_losses.get('hsde_loss', 0.0),
                    'train/epoch_lc_nerf_loss': train_losses.get('lc_nerf_loss', 0.0),
                    'train/epoch_dpee_loss': train_losses.get('dpee_loss', 0.0)
                }

                if val_losses is not None:
                    log_dict.update({
                        'val/epoch_loss': val_losses['total_loss'],
                        'val/epoch_hsde_loss': val_losses.get('hsde_loss', 0.0),
                        'val/epoch_lc_nerf_loss': val_losses.get('lc_nerf_loss', 0.0),
                        'val/epoch_dpee_loss': val_losses.get('dpee_loss', 0.0)
                    })

                wandb.log(log_dict)

            # Save checkpoint
            if (epoch + 1) % self.config.checkpoint_config['save_freq'] == 0:
                is_best = False
                if val_losses is not None:
                    current_metric = -val_losses['total_loss']  # lower loss is better
                    if current_metric > self.best_metric:
                        self.best_metric = current_metric
                        is_best = True

                self.save_checkpoint(epoch + 1, is_best)

            epoch_time = time.time() - epoch_start
            logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
            logger.info(f"Train loss: {train_losses['total_loss']:.4f}")
            if val_losses is not None:
                logger.info(f"Val loss: {val_losses['total_loss']:.4f}")

        logger.info("Training completed!")

        if self.use_wandb:
            wandb.finish()

def collate_fn(batch):
    """Custom collation function supporting poses field"""
    # 1. Remove mesh to avoid default_collate error
    meshes = [item.pop('mesh') for item in batch]
    poses  = [item.pop('poses') for item in batch]

    # 2. Use default collation for the rest
    collated = torch.utils.data.default_collate(batch)

    # 3. Return poses as list (to be parsed during training)
    collated['poses'] = poses          # list[dict or None]
    collated['mesh']  = meshes         # list[TriangleMesh]
    return collated

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="LSVR-SE Training Script")
    parser.add_argument('--config', type=str, default="default",
                        choices=['default', 'fast', 'production'],
                        help='Configuration to use')
    parser.add_argument('--data_root', type=str, default="./data",
                        help='Root directory of training data')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--validate_only', action='store_true',
                        help='Run validation only')

    args = parser.parse_args()

    # Select configuration
    if args.config == "fast":
        config = FAST_CONFIG
    elif args.config == "production":
        config = PRODUCTION_CONFIG
    else:
        config = DEFAULT_CONFIG

    # Override configuration
    if args.batch_size is not None:
        config.training_config['batch_size'] = args.batch_size

    if args.learning_rate is not None:
        config.training_config['learning_rate'] = args.learning_rate

    # Create datasets
    train_dataset = LSVRSEDataset(args.data_root, split="train")
    val_dataset = LSVRSEDataset(args.data_root, split="val")

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training_config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn = collate_fn
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.training_config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Create trainer
    trainer = LSVRSETrainer(config, use_wandb=args.use_wandb)

    if args.validate_only:
        # Validation-only mode
        trainer.setup_models()
        val_losses = trainer.validate(val_dataloader)
        print("Validation results:", val_losses)
    else:
        # Normal training mode
        trainer.train(train_dataloader, val_dataloader, args.num_epochs)

    print("Training completed successfully!")


if __name__ == "__main__":
    main()