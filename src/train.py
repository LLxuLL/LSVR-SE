#!/usr/bin/env python3
"""
LSVR-SE 完整训练脚本
支持HSDE、LC-NeRF、DPEE三个组件的联合训练和单独训练
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'  # 5分钟
os.environ['HF_HUB_ETAG_TIMEOUT'] = '30'  # 30秒

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

# 导入LSVR-SE组件
from lsvr_se_config import LSVRSEConfig, LSVRSEModelManager, DEFAULT_CONFIG, FAST_CONFIG, PRODUCTION_CONFIG
from hsde import HSDE, HSDEConfig, HSDELoss
from lc_nerf import LanguageConditionedNeRF, LCNerfConfig, LCNerfLoss
from dpee import DifferentiableProgrammaticEditingEngine, DPEEConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LSVR-SE-Training")


class LSVRSEDataset(Dataset):
    """LSVR-SE训练数据集"""

    def __init__(self, data_root: str, split: str = "train", transform=None):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform

        # 数据路径
        self.image_dir = self.data_root / split / "images"
        self.mesh_dir = self.data_root / split / "meshes"
        self.text_dir = self.data_root / split / "texts"

        # 确保目录存在
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.mesh_dir.mkdir(parents=True, exist_ok=True)
        self.text_dir.mkdir(parents=True, exist_ok=True)

        # 加载数据列表
        self.data_list = self._load_data_list()

        logger.info(f"Loaded {len(self.data_list)} samples for {split} split")

    def _load_data_list(self) -> List[Dict[str, str]]:
        """加载数据列表"""
        data_list = []

        # 扫描图像文件
        image_files = list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png"))

        for image_file in image_files:
            base_name = image_file.stem

            # 检查对应的网格文件
            mesh_file = self.mesh_dir / f"{base_name}.ply"
            if not mesh_file.exists():
                # 创建默认网格
                mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
                o3d.io.write_triangle_mesh(str(mesh_file), mesh)

            # 检查对应的文本文件
            text_file = self.text_dir / f"{base_name}.txt"
            if not text_file.exists():
                # 创建默认文本
                default_texts = [
                    "添加窗户",
                    "移除门",
                    "旋转45度",
                    "缩放1.5倍",
                    "添加柱子"
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
            # ---- 1. 图像 ----
            image = Image.open(data['image']).convert('RGB')
            image = image.resize((224, 224))
            image_array = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)

            # ---- 2. 文本 ----
            with open(data['text'], 'r', encoding='utf-8') as f:
                text = f.read().strip()

            # ---- 3. 网格（只做占位，后面可重构）----
            mesh = o3d.io.read_triangle_mesh(data['mesh'])
            if not mesh.has_vertices():
                mesh = o3d.geometry.TriangleMesh.create_box()

            # ---- 4. 相机 pose（新增） ----
            pose_file = Path(data['mesh']).parent.parent / 'poses' / f'{base_name}.json'
            if pose_file.exists():
                with open(pose_file, 'r') as f:
                    poses = json.load(f)  # dict 或 list
            else:
                poses = None  # 占位
            return {
                'image': image_tensor,
                'text': text,
                'mesh': mesh,
                'poses': poses,  # 新增字段
                'base_name': base_name
            }
        except Exception as e:
            logger.error(f"Error loading {base_name}: {e}")
            # 返回默认占位
            return {
                'image': torch.randn(3, 224, 224),
                'text': 'placeholder',
                'mesh': o3d.geometry.TriangleMesh.create_box(),
                'poses': None,
                'base_name': 'error'
            }

    def _get_default_sample(self):
        """获取默认样本"""
        image = torch.randn(3, 224, 224)
        mesh = o3d.geometry.TriangleMesh.create_box()
        text = "添加窗户"

        return {
            'image': image,
            'mesh': mesh,
            'text': text,
            'base_name': 'default'
        }


class LSVRSETrainer:
    """LSVR-SE训练器"""

    def __init__(self, config: LSVRSEConfig, use_wandb: bool = True):
        self.config = config
        self.use_wandb = use_wandb

        # 初始化模型管理器
        self.model_manager = LSVRSEModelManager(config)

        # 组件
        self.hsde = None
        self.lc_nerf = None
        self.dpee = None

        # 损失函数
        self.hsde_loss = None
        self.lc_nerf_loss = None

        # 优化器
        self.optimizer = None
        self.scheduler = None
        self.scaler = None

        # 设备
        self.device = torch.device(config.device)

        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0

        # 初始化WandB
        if use_wandb:
            wandb.init(
                project=config.project_name,
                config=config,
                name=f"lsvr_se_train_{int(time.time())}"
            )

    def setup_models(self):
        """设置模型"""
        logger.info("Setting up models...")

        try:
            # 初始化模型管理器
            self.model_manager.initialize_components()

            # 获取组件
            self.hsde = self.model_manager.hsde
            self.lc_nerf = self.model_manager.lc_nerf
            self.dpee = self.model_manager.dpee

            # 初始化损失函数
            if self.hsde is not None:
                hsde_config = HSDEConfig(**self.config.hsde_config)
                self.hsde_loss = HSDELoss(hsde_config)

            if self.lc_nerf is not None:
                lc_nerf_config = LCNerfConfig(**self.config.lc_nerf_config)
                self.lc_nerf_loss = LCNerfLoss(lc_nerf_config)

            # 设置优化器
            self._setup_optimizer()

            # 设置混合精度
            if self.config.mixed_precision:
                self.scaler = GradScaler()

            logger.info("Models setup completed!")

        except Exception as e:
            logger.error(f"Failed to setup models: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _setup_optimizer(self):
        """设置优化器"""
        trainable_params = []

        if self.hsde is not None:
            trainable_params.extend(list(self.hsde.parameters()))

        if self.lc_nerf is not None:
            trainable_params.extend(list(self.lc_nerf.parameters()))

        if self.dpee is not None:
            trainable_params.extend(list(self.dpee.remesh_network.parameters()))

        if not trainable_params:
            raise ValueError("No trainable parameters found!")

        # 创建优化器
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

        # 创建学习率调度器
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
        """训练一个epoch"""
        logger.info(f"Starting epoch {epoch}")

        # 设置训练模式
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
                    # 前向传播
                    losses = self.train_step(batch)

                    # 更新损失
                    for key, value in losses.items():
                        if key in epoch_losses:
                            epoch_losses[key] += value

                    # 更新进度条
                    pbar.set_postfix({
                        'loss': f"{losses['total_loss']:.4f}",
                        'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                    })

                    # 记录到WandB
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

        # 计算平均损失
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single training step: Ensure total_loss is a tensor, and scheduler is called correctly"""
        self.optimizer.zero_grad()

        images = batch['image'].to(self.device)
        texts = batch['text']
        # meshes not used currently, kept empty for future use
        meshes = []

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        losses = {'total_loss': 0.0}

        if self.hsde is not None:
            loss = self._train_hsde_step(images, texts, batch)  # pass batch
            losses['hsde_loss'] = loss.item()
            total_loss = total_loss + loss

        if self.lc_nerf is not None:
            loss = self._train_lc_nerf_step(images, texts, meshes, batch)  # pass batch
            losses['lc_nerf_loss'] = loss.item()
            total_loss = total_loss + loss

        if self.dpee is not None:
            loss = self._train_dpee_step(meshes, texts, batch)  # pass batch
            losses['dpee_loss'] = loss.item()
            total_loss = total_loss + loss

        losses['total_loss'] = total_loss.item()

        # Backward pass and scheduler (unchanged)
        if self.config.mixed_precision and self.scaler is not None:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            self.optimizer.step()

        # Learning rate scheduler must be called after optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return losses

    def _train_hsde_step(self, images: torch.Tensor, texts: List[str]) -> torch.Tensor:
        """HSDE 训练子步骤：保证返回 tensor，且索引为 long"""
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            text_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            input_ids = text_inputs['input_ids'].to(self.device)  # 已是 long
            attention_mask = text_inputs.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            with autocast(enabled=self.config.mixed_precision, device_type='cpu'):
                results = self.hsde(images, input_ids, attention_mask)

            targets = self._create_hsde_targets(results, texts)
            losses = self.hsde_loss(results['predictions'], targets)
            return losses['total_loss']  # 保持 tensor
        except Exception as e:
            logger.error(f"HSDE training step failed: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def _train_lc_nerf_step(self, images: torch.Tensor, texts: List[str], meshes) -> torch.Tensor:
        """LC-NeRF 训练子步骤：兼容 4 维图像"""
        try:
            batch_size, _, height, width = images.shape  # [B,C,H,W]
            rays_o, rays_d = self._generate_training_rays(batch_size, height, width)

            # 随机文本 id
            input_ids = torch.randint(0, 1000, (batch_size, 77), device=self.device)

            with autocast(enabled=self.config.mixed_precision, device_type='cpu'):
                nerf_results = self.lc_nerf(rays_o, rays_d, input_ids)

            targets = self._create_lc_nerf_targets(nerf_results, images)
            losses = self.lc_nerf_loss(nerf_results, targets)
            return losses['total_loss']
        except Exception as e:
            logger.error(f"LC-NeRF training step failed: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def _train_dpee_step(self, meshes: List[o3d.geometry.TriangleMesh], texts: List[str],
                         batch: Dict[str, Any]) -> torch.Tensor:
        """
        DPEE training step: Use real loss from batch if available, otherwise use random constants as placeholders
        batch should contain:
            - edit_success: [B]  bool  Whether edit was successful
            - chamfer_loss: [B]  float  Chamfer distance between before/after edit
            - stability_score: [B] float  Physical stability score
        """
        device = self.device

        # Try to read real loss values
        if 'chamfer_loss' in batch:
            # Real chamfer loss
            loss = batch['chamfer_loss'].mean()
            return loss.requires_grad_(True)

        if 'edit_success' in batch:
            # Use success rate as reward
            success = batch['edit_success'].float()
            loss = 1.0 - success.mean()  # Minimize failure rate
            return loss.requires_grad_(True)

        # Fallback: Random constant loss (consistent with original code behavior)
        total_loss = 0.0
        for text in texts:
            if "添加" in text or "add" in text.lower():
                total_loss += 0.1
            elif "移除" in text or "remove" in text.lower():
                total_loss += 0.2
            else:
                total_loss += 0.05
        denominator = max(len(texts), 1)
        loss = torch.tensor(total_loss / denominator, device=device, requires_grad=True)
        return loss

    def _generate_training_rays(self, batch_size: int, height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成训练用的射线"""
        # 简化的射线生成
        num_rays = height * width

        rays_o = torch.zeros(batch_size, num_rays, 3, device=self.device)
        rays_d = torch.randn(batch_size, num_rays, 3, device=self.device)
        rays_d = F.normalize(rays_d, dim=-1)

        return rays_o, rays_d

    def _create_hsde_targets(self, results: Dict[str, Any], batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        HSDE training targets: Use real values from batch if available, otherwise use random placeholders
        batch should contain:
            - semantic_labels: [B, N]  long
            - bbox_targets:    [B, N, 6]  float
            - confidence_targets: [B, N]  float
        """
        batch_size, num_anchors = results['fused_features'].shape[:2]
        device = self.device

        # Try to read real values (if dataset provides them)
        if 'semantic_labels' in batch:
            semantic_labels = batch['semantic_labels']
            bbox_targets = batch['bbox_targets']
            confidence_targets = batch['confidence_targets']
        else:
            # Fallback: Random generation (consistent with original code behavior)
            semantic_labels = torch.randint(0, 128, (batch_size, num_anchors), device=device)
            bbox_targets = torch.randn(batch_size, num_anchors, 6, device=device)
            confidence_targets = torch.rand(batch_size, num_anchors, device=device)

        return {
            'semantic_labels': semantic_labels,
            'bbox_targets': bbox_targets,
            'confidence_targets': confidence_targets
        }

    def _create_lc_nerf_targets(self, results: Dict[str, Any], batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        LC-NeRF training targets: Use real values from batch if available, otherwise use random placeholders
        batch should contain:
            - rgb_gt: [B, H, W, 3]  float  Real multi-view RGB
            - depth_gt: [B, H, W]   float  Real depth maps
            - semantic_targets: [B, 512] float  Text embedding ground truth
        """
        batch_size, height, width = results['rgb_map'].shape[:3]
        device = self.device

        # Try to read real values
        if 'rgb_gt' in batch:
            rgb_gt = batch['rgb_gt']
            # If batch format is [B, H, W, 3], use directly
            # If format is [B, 3, H, W], need to permute
            if rgb_gt.dim() == 4 and rgb_gt.shape[1] == 3:
                rgb_gt = rgb_gt.permute(0, 2, 3, 1)
        else:
            # Fallback: Random RGB
            rgb_gt = torch.rand(batch_size, height, width, 3, device=device)

        if 'semantic_targets' in batch:
            semantic_targets = batch['semantic_targets']
        else:
            # Fallback: Random text embeddings
            semantic_targets = torch.randn(batch_size, 512, device=device)

        return {
            'rgb_gt': rgb_gt,
            'semantic_targets': semantic_targets
        }

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """验证模型"""
        logger.info("Starting validation...")

        # 设置评估模式
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
                        # 前向传播
                        losses = self.train_step(batch)

                        # 更新损失
                        for key, value in losses.items():
                            if key in val_losses:
                                val_losses[key] += value

                        pbar.set_postfix({'val_loss': f"{losses['total_loss']:.4f}"})

                    except Exception as e:
                        logger.error(f"Error in validation batch: {str(e)}")
                        continue

        # 计算平均损失
        for key in val_losses:
            val_losses[key] /= num_batches

        return val_losses

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
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
        """主训练循环"""
        logger.info("Starting training...")

        # 设置模型
        self.setup_models()

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # 训练一个epoch
            train_losses = self.train_epoch(train_dataloader, epoch)

            # 验证
            val_losses = None
            if val_dataloader is not None:
                val_losses = self.validate(val_dataloader)

            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step()

            # 记录到WandB
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

            # 保存检查点
            if (epoch + 1) % self.config.checkpoint_config['save_freq'] == 0:
                is_best = False
                if val_losses is not None:
                    current_metric = -val_losses['total_loss']  # 损失越小越好
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
    """支持 poses 字段的自定义打包"""
    # 1. 弹出 mesh 避免 default_collate 报错
    meshes = [item.pop('mesh') for item in batch]
    poses  = [item.pop('poses') for item in batch]

    # 2. 其余用默认打包
    collated = torch.utils.data.default_collate(batch)

    # 3. 把 poses 按列表原样返回（后续训练自己解析）
    collated['poses'] = poses          # list[dict or None]
    collated['mesh']  = meshes         # list[TriangleMesh]
    return collated

def main():
    """主函数"""
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

    # 选择配置
    if args.config == "fast":
        config = FAST_CONFIG
    elif args.config == "production":
        config = PRODUCTION_CONFIG
    else:
        config = DEFAULT_CONFIG

    # 覆盖配置
    if args.batch_size is not None:
        config.training_config['batch_size'] = args.batch_size

    if args.learning_rate is not None:
        config.training_config['learning_rate'] = args.learning_rate

    # 创建数据集
    train_dataset = LSVRSEDataset(args.data_root, split="train")
    val_dataset = LSVRSEDataset(args.data_root, split="val")

    # 创建数据加载器
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

    # 创建训练器
    trainer = LSVRSETrainer(config, use_wandb=args.use_wandb)

    if args.validate_only:
        # 仅验证模式
        trainer.setup_models()
        val_losses = trainer.validate(val_dataloader)
        print("Validation results:", val_losses)
    else:
        # 正常训练模式
        trainer.train(train_dataloader, val_dataloader, args.num_epochs)

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
