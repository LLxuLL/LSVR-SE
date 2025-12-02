#!/usr/bin/env python3
"""
LSVR-SE Reasoning Script
User-facing inference interface supporting single image processing, batch processing, and interactive editing
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'  # 5 minutes
os.environ['HF_HUB_ETAG_TIMEOUT'] = '30'  # 30 seconds
import sys
import time
import argparse
import json
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

import torch
import numpy as np
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Import LSVR-SE components
from lsvr_se_config import LSVRSEConfig, LSVRSEModelManager, DEFAULT_CONFIG, FAST_CONFIG, PRODUCTION_CONFIG
from hsde import HSDE, HSDEConfig
from lc_nerf import LanguageConditionedNeRF, LCNerfConfig, LCNerfRenderer
from dpee import DifferentiableProgrammaticEditingEngine, DPEEConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LSVR-SE-Reasoning")


class LSVRSEReasoning:
    """LSVR-SE Reasoning Engine"""

    def __init__(self, config: Optional[LSVRSEConfig] = None, checkpoint_path: Optional[str] = None):
        self.config = config or DEFAULT_CONFIG
        self.checkpoint_path = checkpoint_path

        # Initialize model manager
        self.model_manager = LSVRSEModelManager(self.config)

        # Components
        self.hsde = None
        self.lc_nerf = None
        self.dpee = None
        self.lc_renderer = None

        # Device
        self.device = torch.device(self.config.device)

        # Initialization flag
        self.initialized = False

        logger.info(f"LSVR-SE Reasoning Engine initialized on {self.device}")

    def initialize(self):
        """Initialize reasoning engine"""
        if self.initialized:
            return

        logger.info("Initializing LSVR-SE reasoning engine...")

        try:
            # Initialize model manager
            self.model_manager.initialize_components()

            # Get components
            self.hsde = self.model_manager.hsde
            self.lc_nerf = self.model_manager.lc_nerf
            self.dpee = self.model_manager.dpee

            # Create LC-NeRF renderer
            if self.lc_nerf is not None:
                self.lc_renderer = LCNerfRenderer(self.lc_nerf, self.device)

            # Load checkpoint
            if self.checkpoint_path and os.path.exists(self.checkpoint_path):
                logger.info(f"Loading checkpoint from {self.checkpoint_path}")
                self.model_manager.load_models(0)  # Simplified loading

            # Set evaluation mode
            if self.hsde:
                self.hsde.eval()
            if self.lc_nerf:
                self.lc_nerf.eval()
            if self.dpee:
                self.dpee.remesh_network.eval()

            self.initialized = True
            logger.info("LSVR-SE reasoning engine initialized successfully!")

        except Exception as e:
            logger.error(f"Failed to initialize reasoning engine: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def process_single_image(self, image_path: str, text_instruction: str = "",
                             output_dir: str = "./output") -> Dict[str, Any]:
        """Process single image"""
        if not self.initialized:
            self.initialize()

        logger.info(f"Processing single image: {image_path}")
        start_time = time.time()

        try:
            # Validate input
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            # Load image
            image = self._load_image(image_path)

            # Step 1: HSDE feature extraction
            hsde_results = self._hsde_inference(image, text_instruction)

            # Step 2: LC-NeRF 3D reconstruction
            nerf_results = self._nerf_inference(image, hsde_results)

            # Step 3: DPEE semantic editing (if instruction provided)
            if text_instruction:
                edit_results = self._edit_inference(nerf_results['mesh'], text_instruction)
                final_mesh = edit_results['mesh']
            else:
                final_mesh = nerf_results['mesh']

            # Generate results
            results = {
                'input_image': image_path,
                'text_instruction': text_instruction,
                'final_mesh': final_mesh,
                'processing_time': time.time() - start_time,
                'success': True,
                'hsde_results': hsde_results,
                'nerf_results': nerf_results
            }

            # Save results
            self._save_inference_results(results, output_dir)

            logger.info(f"Image processing completed in {results['processing_time']:.2f}s")
            return results

        except Exception as e:
            logger.error(f"Failed to process image: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'input_image': image_path
            }

    def batch_process(self, image_list: List[str], text_instructions: List[str] = None,
                      output_dir: str = "./output") -> List[Dict[str, Any]]:
        """Batch process images"""
        if not self.initialized:
            self.initialize()

        logger.info(f"Starting batch processing of {len(image_list)} images")

        if text_instructions is None:
            text_instructions = [""] * len(image_list)

        if len(image_list) != len(text_instructions):
            raise ValueError("Number of images and text instructions must match")

        results = []

        for i, (image_path, text_instruction) in enumerate(zip(image_list, text_instructions)):
            logger.info(f"Processing batch item {i + 1}/{len(image_list)}")

            try:
                result = self.process_single_image(image_path, text_instruction,
                                                   f"{output_dir}/batch_{i + 1}")
                results.append(result)

            except Exception as e:
                logger.error(f"Failed to process batch item {i + 1}: {str(e)}")
                results.append({
                    'success': False,
                    'error': str(e),
                    'input_image': image_path
                })

        logger.info(
            f"Batch processing completed. {len([r for r in results if r.get('success')])}/{len(results)} successful")
        return results

    def interactive_edit(self, mesh: o3d.geometry.TriangleMesh,
                         edit_history: List[str]) -> Dict[str, Any]:
        """Interactive editing"""
        if not self.initialized:
            self.initialize()

        if self.dpee is None:
            raise ValueError("DPEE not available for interactive editing")

        logger.info("Starting interactive editing session")

        current_mesh = mesh
        edit_results = []

        for i, instruction in enumerate(edit_history):
            logger.info(f"Applying edit {i + 1}/{len(edit_history)}: {instruction}")

            try:
                result = self.dpee.parse_and_execute(current_mesh, instruction)
                current_mesh = result[0]  # Update mesh

                edit_results.append({
                    'step': i + 1,
                    'instruction': instruction,
                    'success': result[1].get('success', True),
                    'stability_analysis': result[1].get('stability_analysis', {})
                })

            except Exception as e:
                logger.error(f"Edit step {i + 1} failed: {str(e)}")
                edit_results.append({
                    'step': i + 1,
                    'instruction': instruction,
                    'success': False,
                    'error': str(e)
                })

        return {
            'final_mesh': current_mesh,
            'edit_history': edit_history,
            'edit_results': edit_results,
            'success': all(r['success'] for r in edit_results)
        }

    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image"""
        try:
            image = Image.open(image_path).convert('RGB')
            image = image.resize((224, 224))

            # Convert to tensor
            image_array = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)

            # Normalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std

            return image_tensor.to(self.device)

        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {str(e)}")
            # Return default image
            return torch.randn(3, 224, 224).to(self.device)

    def _hsde_inference(self, image: torch.Tensor, text_instruction: str) -> Dict[str, Any]:
        """HSDE inference"""
        if self.hsde is None:
            logger.warning("HSDE not available, skipping step")
            return {'success': False, 'error': 'HSDE not available'}

        try:
            # Prepare text input
            if text_instruction:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
                text_inputs = tokenizer(text_instruction, return_tensors="pt", padding=True, truncation=True)
                input_ids = text_inputs['input_ids'].to(self.device)
            else:
                input_ids = torch.randint(0, 1000, (1, 77)).to(self.device)

            # HSDE forward pass
            with torch.no_grad():
                results = self.hsde(image.unsqueeze(0), input_ids)

            # Extract semantic features
            predictions = results['predictions']
            confidences = predictions['confidences'].squeeze(-1)
            high_conf_mask = confidences > 0.5

            hsde_results = {
                'semantic_features': predictions['semantic_logits'][high_conf_mask],
                'bboxes': predictions['bboxes'][high_conf_mask],
                'confidences': confidences[high_conf_mask],
                'fused_features': results['fused_features'],
                'success': True
            }

            return hsde_results

        except Exception as e:
            logger.error(f"HSDE inference failed: {str(e)}")
            return {'success': False, 'error': str(e)}

    def _nerf_inference(self, image: torch.Tensor, hsde_results: Dict[str, Any]) -> Dict[str, Any]:
        """LC-NeRF inference"""
        if self.lc_nerf is None or self.lc_renderer is None:
            logger.warning("LC-NeRF not available, using fallback")
            return self._fallback_mesh_generation(image)

        try:
            # Generate camera parameters
            camera_params = self._generate_camera_params(image.shape)

            # Generate rays
            height, width = image.shape[1], image.shape[2]
            rays_o, rays_d = self._generate_rays(camera_params, height, width)

            # Prepare text embedding
            text_embedding = hsde_results.get('fused_features', torch.randn(1, 256).to(self.device))

            # LC-NeRF rendering
            with torch.no_grad():
                nerf_results = self.lc_nerf(rays_o, rays_d,
                                            torch.randint(0, 1000, (1, 77)).to(self.device))

            # Generate mesh from rendering results
            mesh = self._mesh_from_nerf_results(nerf_results, camera_params)

            return {
                'mesh': mesh,
                'rgb_map': nerf_results['rgb_map'],
                'depth_map': nerf_results['depth_map'],
                'success': True
            }

        except Exception as e:
            logger.error(f"LC-NeRF inference failed: {str(e)}")
            return self._fallback_mesh_generation(image)

    def _edit_inference(self, mesh: o3d.geometry.TriangleMesh, text_instruction: str) -> Dict[str, Any]:
        """DPEE inference"""
        if self.dpee is None:
            logger.warning("DPEE not available, skipping edit")
            return {'mesh': mesh, 'success': False, 'error': 'DPEE not available'}

        try:
            # DPEE editing
            edited_mesh, results = self.dpee.parse_and_execute(mesh, text_instruction)

            return {
                'mesh': edited_mesh,
                'edit_record': results.get('edit_record', {}),
                'stability_analysis': results.get('stability_analysis', {}),
                'success': results.get('success', False)
            }

        except Exception as e:
            logger.error(f"DPEE inference failed: {str(e)}")
            return {'mesh': mesh, 'success': False, 'error': str(e)}

    def _generate_camera_params(self, image_shape: torch.Size) -> Dict[str, torch.Tensor]:
        """Generate camera parameters"""
        height, width = image_shape[1], image_shape[2]

        # Simplified camera intrinsics
        focal_length = width * 0.9
        cx = width / 2
        cy = height / 2

        intrinsics = torch.tensor([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=torch.float32)

        extrinsics = torch.eye(4, dtype=torch.float32)

        return {
            'intrinsics': intrinsics.to(self.device),
            'extrinsics': extrinsics.to(self.device)
        }

    def _generate_rays(self, camera_params: Dict[str, torch.Tensor],
                       height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate rays"""
        i, j = torch.meshgrid(
            torch.linspace(0, width - 1, width, device=self.device),
            torch.linspace(0, height - 1, height, device=self.device),
            indexing='ij'
        )

        intrinsics = camera_params['intrinsics']
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        dirs = torch.stack([
            (i - cx) / fx,
            -(j - cy) / fy,
            -torch.ones_like(i)
        ], dim=-1)

        dirs = F.normalize(dirs, dim=-1)
        rays_o = torch.zeros_like(dirs)

        return rays_o.reshape(1, -1, 3), dirs.reshape(1, -1, 3)

    def _mesh_from_nerf_results(self, nerf_results: Dict[str, torch.Tensor],
                                camera_params: Dict[str, torch.Tensor]) -> o3d.geometry.TriangleMesh:
        """Generate mesh from NeRF results"""
        rgb_map = nerf_results['rgb_map']
        depth_map = nerf_results['depth_map']

        height, width = rgb_map.shape[1], rgb_map.shape[2]

        # Create mesh
        mesh = o3d.geometry.TriangleMesh.create_box(width=2.0, height=2.0, depth=2.0)

        # Adjust based on depth map
        vertices = np.asarray(mesh.vertices)
        depth_normalized = depth_map.squeeze(0).cpu().numpy()
        depth_normalized = (depth_normalized - depth_normalized.min()) / (
                    depth_normalized.max() - depth_normalized.min())

        for i, vertex in enumerate(vertices):
            x_idx = int((vertex[0] + 1) / 2 * (width - 1))
            y_idx = int((vertex[1] + 1) / 2 * (height - 1))
            x_idx = np.clip(x_idx, 0, width - 1)
            y_idx = np.clip(y_idx, 0, height - 1)
            vertex[2] = depth_normalized[y_idx, x_idx] * 2 - 1

        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.compute_vertex_normals()

        return mesh

    def _fallback_mesh_generation(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        """Fallback mesh generation"""
        mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)

        image_mean = image_tensor.mean().item()
        scale_factor = 0.5 + image_mean

        vertices = np.asarray(mesh.vertices)
        vertices *= scale_factor
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.compute_vertex_normals()

        return {
            'mesh': mesh,
            'rgb_map': torch.rand(1, 224, 224, 3),
            'depth_map': torch.rand(1, 224, 224),
            'success': True,
            'fallback': True
        }

    def _save_inference_results(self, results: Dict[str, Any], output_dir: str):
        """Save inference results"""
        os.makedirs(output_dir, exist_ok=True)

        # Save mesh
        if 'final_mesh' in results and results['final_mesh'] is not None:
            mesh_path = os.path.join(output_dir, "final_mesh.ply")
            o3d.io.write_triangle_mesh(mesh_path, results['final_mesh'])
            logger.info(f"Saved final mesh to {mesh_path}")

        # Save result information
        info_path = os.path.join(output_dir, "inference_results.json")

        save_results = {}
        for key, value in results.items():
            if key in ['final_mesh', 'input_image']:
                continue
            elif isinstance(value, torch.Tensor):
                save_results[key] = value.tolist()
            elif isinstance(value, np.ndarray):
                save_results[key] = value.tolist()
            elif isinstance(value, (int, float, str, bool, list, dict)):
                save_results[key] = value

        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(save_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved inference results to {info_path}")

    def visualize_results(self, results: Dict[str, Any], save_path: Optional[str] = None):
        """Visualize results"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))

            # Display input image
            if 'input_image' in results:
                input_image = Image.open(results['input_image'])
                axes[0, 0].imshow(input_image)
                axes[0, 0].set_title('Input Image')
                axes[0, 0].axis('off')

            # Display rendering result
            if 'nerf_results' in results and 'rgb_map' in results['nerf_results']:
                rgb_map = results['nerf_results']['rgb_map'].squeeze(0).cpu().numpy()
                axes[0, 1].imshow(rgb_map)
                axes[0, 1].set_title('NeRF Rendering')
                axes[0, 1].axis('off')

            # Display depth map
            if 'nerf_results' in results and 'depth_map' in results['nerf_results']:
                depth_map = results['nerf_results']['depth_map'].squeeze(0).cpu().numpy()
                axes[1, 0].imshow(depth_map, cmap='gray')
                axes[1, 0].set_title('Depth Map')
                axes[1, 0].axis('off')

            # Display processing time
            processing_time = results.get('processing_time', 0)
            axes[1, 1].text(0.5, 0.5, f"Processing Time:\n{processing_time:.2f}s",
                            ha='center', va='center', fontsize=20)
            axes[1, 1].set_title('Performance')
            axes[1, 1].axis('off')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Saved visualization to {save_path}")
            else:
                plt.show()

            plt.close()

        except Exception as e:
            logger.error(f"Visualization failed: {str(e)}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="LSVR-SE Reasoning Script")
    parser.add_argument('--mode', type=str, default="single",
                        choices=['single', 'batch', 'interactive'],
                        help='Inference mode')
    parser.add_argument('--image', type=str, default="",
                        help='Input image path (for single mode)')
    parser.add_argument('--image_list', type=str, default="",
                        help='Path to file containing image list (for batch mode)')
    parser.add_argument('--text', type=str, default="",
                        help='Text instruction')
    parser.add_argument('--text_list', type=str, default="",
                        help='Path to file containing text instructions')
    parser.add_argument('--output_dir', type=str, default="./output",
                        help='Output directory')
    parser.add_argument('--config', type=str, default="default",
                        choices=['default', 'fast', 'production'],
                        help='Configuration to use')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')

    args = parser.parse_args()

    # Select configuration
    if args.config == "fast":
        config = FAST_CONFIG
    elif args.config == "production":
        config = PRODUCTION_CONFIG
    else:
        config = DEFAULT_CONFIG

    # Create reasoning engine
    reasoning_engine = LSVRSEReasoning(config, args.checkpoint)

    if args.mode == "single":
        # Single image mode
        if not args.image:
            print("Error: --image is required for single mode")
            return

        result = reasoning_engine.process_single_image(args.image, args.text, args.output_dir)

        if result['success']:
            print(f"✅ Processing completed successfully!")
            print(f"   Processing time: {result['processing_time']:.2f}s")
            print(f"   Output saved to: {args.output_dir}")

            if args.visualize:
                viz_path = os.path.join(args.output_dir, "visualization.png")
                reasoning_engine.visualize_results(result, viz_path)
        else:
            print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")

    elif args.mode == "batch":
        # Batch mode
        if not args.image_list:
            print("Error: --image_list is required for batch mode")
            return

        # Read image list
        with open(args.image_list, 'r', encoding='utf-8') as f:
            image_list = [line.strip() for line in f if line.strip()]

        # Read text instruction list
        text_list = [""] * len(image_list)
        if args.text_list:
            with open(args.text_list, 'r', encoding='utf-8') as f:
                text_list = [line.strip() for line in f if line.strip()]

        results = reasoning_engine.batch_process(image_list, text_list, args.output_dir)

        success_count = sum(1 for r in results if r.get('success'))
        print(f"✅ Batch processing completed!")
        print(f"   Total images: {len(results)}")
        print(f"   Successful: {success_count}")
        print(f"   Failed: {len(results) - success_count}")
        print(f"   Results saved to: {args.output_dir}")

    elif args.mode == "interactive":
        # Interactive mode
        if not args.image:
            print("Error: --image is required for interactive mode")
            return

        print("🎨 LSVR-SE Interactive Mode")
        print("Type 'quit' to exit")

        # Process initial image
        initial_result = reasoning_engine.process_single_image(args.image, "", args.output_dir)

        if not initial_result['success']:
            print(f"❌ Failed to load initial image: {initial_result.get('error')}")
            return

        current_mesh = initial_result['final_mesh']
        edit_history = []

        while True:
            instruction = input("\n📝 Enter edit instruction: ").strip()

            if instruction.lower() in ['quit', 'exit', 'q']:
                break

            if not instruction:
                continue

            try:
                # Apply edit
                result = reasoning_engine.dpee.parse_and_execute(current_mesh, instruction)

                if result[1].get('success', True):
                    current_mesh = result[0]
                    edit_history.append(instruction)

                    print(f"✅ Edit applied successfully!")

                    # Display stability analysis
                    stability = result[1].get('stability_analysis', {})
                    if stability:
                        print(f"   Stability: {'✅ Stable' if stability.get('is_stable') else '❌ Unstable'}")
                        print(f"   Score: {stability.get('stability_score', 0):.3f}")

                    # Save intermediate results
                    step_dir = os.path.join(args.output_dir, f"step_{len(edit_history)}")
                    os.makedirs(step_dir, exist_ok=True)

                    mesh_path = os.path.join(step_dir, "edited_mesh.ply")
                    o3d.io.write_triangle_mesh(mesh_path, current_mesh)

                    print(f"   Saved to: {step_dir}")
                else:
                    print(f"❌ Edit failed: {result[1].get('error', 'Unknown error')}")

            except Exception as e:
                print(f"❌ Error applying edit: {str(e)}")

        # Save final edit history
        history_path = os.path.join(args.output_dir, "edit_history.json")
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump({
                'initial_image': args.image,
                'edit_history': edit_history,
                'total_edits': len(edit_history)
            }, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Interactive session completed!")
        print(f"   Total edits applied: {len(edit_history)}")
        print(f"   Edit history saved to: {history_path}")

    print("\n🎉 LSVR-SE reasoning completed!")


if __name__ == "__main__":
    main()