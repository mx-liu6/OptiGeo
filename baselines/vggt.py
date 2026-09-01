
import os
import sys
from typing import *

import click
import torch
import torch.nn.functional as F
import numpy as np

# Add VGGT module path to Python path
vggt_path = os.environ.get("VGGT_ROOT", "../vggt")
if vggt_path not in sys.path:
    sys.path.insert(0, vggt_path)

from optigeo.test.baseline import OptiGeoBaselineInterface


class Baseline(OptiGeoBaselineInterface):

    def __init__(self, pretrained_model_name_or_path: str, use_fp16: bool, device: str = 'cuda:0'):
        super().__init__()
        # Import VGGT model
        try:
            from vggt.models.vggt import VGGT
            from vggt.utils.load_fn import load_and_preprocess_images
            from vggt.utils.geometry import depth_to_world_coords_points
        except ImportError:
            raise ImportError(
                "Could not import VGGT model. Make sure the VGGT module is in your Python path. "
                "You may need to add the path to the VGGT module to your PYTHONPATH environment variable."
            )

        # Load the VGGT model from local checkpoint
        self.model = VGGT()
        checkpoint = torch.load(pretrained_model_name_or_path, map_location=device)
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(device).eval()

        self.device = torch.device(device)
        self.use_fp16 = use_fp16
        # Determine if we can use bfloat16 based on GPU capability
        if torch.cuda.is_available():
            self.use_bfloat16 = torch.cuda.get_device_capability()[0] >= 8
        else:
            self.use_bfloat16 = False

        # Set the dtype based on GPU capability and use_fp16 flag
        if self.use_fp16:
            self.dtype = torch.bfloat16 if self.use_bfloat16 else torch.float16
        else:
            self.dtype = torch.float32

        # Save the utility functions
        self.preprocess_fn = load_and_preprocess_images
        self.depth_to_world_coords_points = depth_to_world_coords_points

    @click.command()
    @click.option('--pretrained', 'pretrained_model_name_or_path', type=str, default="pretrained/model_vggt.pt", help='Path to the pretrained VGGT model checkpoint')
    @click.option('--fp16', 'use_fp16', is_flag=True, help='Use mixed precision for inference')
    @click.option('--device', type=str, default='cuda:0', help='Device to run inference on')
    @staticmethod
    def load(pretrained_model_name_or_path: str, use_fp16: bool, device: str = 'cuda:0'):
        return Baseline(pretrained_model_name_or_path, use_fp16, device)

    @torch.inference_mode()
    def infer(self, image: torch.FloatTensor, intrinsics: Optional[torch.FloatTensor] = None):
        # Handle single image vs batch
        is_single = image.dim() == 3
        if is_single:
            image = image.unsqueeze(0)

        # Get original image dimensions
        _, _, H, W = image.shape

        # Convert image to numpy and then back through the VGGT preprocessing
        image_np = (image[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Save image to a temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png') as temp_file:
            import cv2
            cv2.imwrite(temp_file.name, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

            # Use VGGT's preprocessing function
            processed_image = self.preprocess_fn([temp_file.name]).to(self.device)

        # Run inference with VGGT model
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.use_fp16, dtype=self.dtype):
                # Add batch dimension if needed
                if processed_image.dim() == 3:
                    processed_image = processed_image.unsqueeze(0)

                # Add sequence dimension - VGGT expects [B, S, C, H, W]
                if processed_image.dim() == 4:
                    processed_image = processed_image.unsqueeze(1)

                # Get predictions from VGGT model
                predictions = self.model(processed_image)

        try:
            # Use VGGT's aggregator to get tokens and ps_idx
            aggregated_tokens_list, ps_idx = self.model.aggregator(processed_image)

            # Predict Depth Maps
            depth_map, depth_conf = self.model.depth_head(aggregated_tokens_list, processed_image, ps_idx)

            # Predict Point Maps directly using point_head
            point_map, point_conf = self.model.point_head(aggregated_tokens_list, processed_image, ps_idx)

            # Extract depth prediction
            depth_pred = depth_map.squeeze()

            # Extract point map
            points = point_map.squeeze()

            # Resize depth to original dimensions if needed
            if depth_pred.shape != (H, W):
                depth_pred = F.interpolate(
                    depth_pred.unsqueeze(0).unsqueeze(0),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()

            # Process point map
            # Ensure points has the right shape
            if points.dim() > 3:
                points = points.squeeze(0)

            # Check if points has shape [3, H, W] and convert to [H, W, 3]
            if points.shape[0] == 3 and points.dim() == 3:
                points = points.permute(1, 2, 0)  # [3, H, W] -> [H, W, 3]

            # Resize points to match original image size if needed
            if points.shape[:2] != (H, W):
                points = F.interpolate(
                    points.permute(2, 0, 1).unsqueeze(0),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).permute(1, 2, 0)

            # Return both depth and points as scale-invariant
            return {
                'depth_scale_invariant': depth_pred,
                'points_scale_invariant': points
            }
        except Exception as e:
            print(f"Error generating point cloud: {e}")
            # If point cloud generation fails, return only depth
            return {
                'depth_scale_invariant': depth_pred
            }

    @torch.inference_mode()
    def infer_for_evaluation(self, image: torch.FloatTensor, intrinsics: torch.FloatTensor = None):
        # For evaluation, we use the same inference method
        return self.infer(image, intrinsics)
