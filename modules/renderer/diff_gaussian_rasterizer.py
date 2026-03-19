"""
Differentiable Gaussian Rasterizer with Language Features
"""

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from dataclasses import dataclass

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)


@dataclass
class RendererConfig:
    """Configuration for Gaussian renderer"""
    debug: bool = False
    invert_bg_prob: float = 1.0
    back_ground_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)


class DiffGaussianRenderer(nn.Module):
    """
    Differentiable Gaussian Rasterizer
    Renders RGB and language features using diff-gaussian-rasterization
    """
    
    def __init__(self, geometry, cfg: RendererConfig = None):
        super().__init__()
        self.geometry = geometry
        self.cfg = cfg or RendererConfig()
        
        self.background_tensor = torch.tensor(
            self.cfg.back_ground_color, dtype=torch.float32, device="cuda"
        )
        
        self.training = True
        
    
    def forward(
        self,
        viewpoint_camera,
        bg_color: torch.Tensor = None,
        scaling_modifier: float = 1.0,
        override_color=None,
        include_feature: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Render the scene
        
        Args:
            viewpoint_camera: Camera parameters
            bg_color: Background color tensor (on GPU)
            scaling_modifier: Scale modifier for Gaussians
            override_color: Override precomputed colors
            include_feature: Whether to render language features
        
        Returns:
            Dict with 'render', 'lang', 'viewspace_points', 'visibility_filter', 'radii'
        """
        if bg_color is None:
            bg_color = self.background_tensor
        
        # Random background inversion during training
        if self.training:
            invert_bg_color = np.random.rand() > self.cfg.invert_bg_prob
        else:
            invert_bg_color = True
        
        bg_color = bg_color if not invert_bg_color else (1.0 - bg_color)
        
        pc = self.geometry
        
        # Create zero tensor for gradient tracking
        screenspace_points = (
            torch.zeros_like(
                pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
            ) + 0
        )
        try:
            screenspace_points.retain_grad()
        except Exception:
            pass
        
        # Setup rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=self.cfg.debug,
            include_feature=include_feature
        )
        
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity
        scales = pc.get_scaling
        rotations = pc.get_rotation
        
        # Colors: SH or precomputed
        shs = None
        colors_precomp = None
        if override_color is None:
            shs = pc.get_features
        else:
            colors_precomp = override_color
        
        # Language features: Normalize to match 3DitScene implementation
        # This ensures the features are unit vectors, which is better for cosine-based semantics
        language_feature_precomp = pc.get_language_feature
        language_feature_precomp = language_feature_precomp / (language_feature_precomp.norm(dim=-1, keepdim=True) + 1e-9)
        
        # Rasterize visible Gaussians to image
        result_list = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            language_feature_precomp=language_feature_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )
        
        rendered_image = result_list[0]
        rendered_feature = result_list[1]
        radii = result_list[2]
        
        # Retain gradients
        if self.training:
            try:
                screenspace_points.retain_grad()
            except Exception:
                pass
        
        return {
            "render": rendered_image.clamp(0, 1),
            "lang": rendered_feature,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }
