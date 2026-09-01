import os
import sys
from pathlib import Path
from typing import *

_script_dir = str(Path(__file__).absolute().parent)
_package_root = str(Path(__file__).absolute().parents[1])
sys.path = [p for p in sys.path if p != _script_dir]
if _package_root not in sys.path:
    sys.path.insert(0, _package_root)

import click
import torch
import utils3d

from optigeo.test.baseline import OptiGeoBaselineInterface

DEFAULT_PRETRAINED_MODEL = "mxliu-hku/OptiGeo"


class Baseline(OptiGeoBaselineInterface):

    def __init__(self, num_tokens: int, resolution_level: int, pretrained_model_name_or_path: str, use_fp16: bool, device: str = 'cuda:0'):
        super().__init__()
        from optigeo.model import import_model_class_by_version
        OptiGeo = import_model_class_by_version()

        self.model = OptiGeo.from_pretrained(pretrained_model_name_or_path).to(device).eval()
        
        self.device = torch.device(device)
        self.num_tokens = num_tokens
        self.resolution_level = resolution_level
        self.use_fp16 = use_fp16
    
    @click.command()
    @click.option('--num_tokens', type=int, default=None)
    @click.option('--resolution_level', type=int, default=9)
    @click.option('--pretrained', 'pretrained_model_name_or_path', type=str, default=DEFAULT_PRETRAINED_MODEL)
    @click.option('--fp16', 'use_fp16', is_flag=True)
    @click.option('--device', type=str, default='cuda:0')
    @staticmethod
    def load(num_tokens: int, resolution_level: int, pretrained_model_name_or_path: str, use_fp16: bool, device: str = 'cuda:0'):
        return Baseline(num_tokens, resolution_level, pretrained_model_name_or_path, use_fp16, device)

    # Implementation for inference
    @torch.inference_mode()
    def infer(self, image: torch.FloatTensor, intrinsics: Optional[torch.FloatTensor] = None):
        if intrinsics is not None:
            fov_x, _ = utils3d.torch.intrinsics_to_fov(intrinsics)
            fov_x = torch.rad2deg(fov_x)
        else:
            fov_x = None
        output = self.model.infer(image, fov_x=fov_x, apply_mask=True, num_tokens=self.num_tokens)
        
        return {
            'points_metric': output['points'],
            'depth_metric': output['depth'],
            'intrinsics': output['intrinsics'],
            'mask': output['mask'],
        }

    @torch.inference_mode()
    def infer_for_evaluation(self, image: torch.FloatTensor, intrinsics: torch.FloatTensor = None):
        if intrinsics is not None:
            fov_x, _ = utils3d.torch.intrinsics_to_fov(intrinsics)
            fov_x = torch.rad2deg(fov_x)
        else:
            fov_x = None
        output = self.model.infer(image, fov_x=fov_x, apply_mask=False, num_tokens=self.num_tokens, use_fp16=self.use_fp16)

        return {
            'points_metric': output['points'],
            'depth_metric': output['depth'],
            'intrinsics': output['intrinsics'],
            'mask': output['mask'],
        }
        
