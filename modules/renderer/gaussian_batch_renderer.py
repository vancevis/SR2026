"""
Gaussian Batch Renderer
Handles batch rendering for multiple views
"""

import torch
from typing import Dict, Any
from dataclasses import dataclass
from .camera import Camera, get_cam_info_gaussian


class GaussianBatchRenderer:
    """
    Batch renderer mixin for multiple camera views.
    Processes multiple views and aggregates outputs.
    Designed to be combined with DiffGaussianRenderer via
    DiffGaussianBatchRenderer.
    """
    
    def batch_forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render multiple views in a batch
        
        Args:
            batch: Dict containing 'c2w', 'fovy', 'width', 'height'
        
        Returns:
            Dict with batched outputs: 'comp_rgb', 'lang', etc.
        """
        bs = batch["c2w"].shape[0]
        
        renders = []
        viewspace_points = []
        visibility_filters = []
        radiis = []
        normals = []
        pred_normals = []
        depths = []
        masks = []
        langs = []
        
        for batch_idx in range(bs):
            batch["batch_idx"] = batch_idx
            fovy = batch["fovy"][batch_idx]
            fovx = batch["fovx"][batch_idx] if "fovx" in batch else fovy
            
            # Get camera info
            w2c, proj, cam_p, cam_proj = get_cam_info_gaussian(
                c2w=batch["c2w"][batch_idx],
                fovx=fovx,
                fovy=fovy,
                znear=0.1,
                zfar=100
            )
            
            # Create camera
            viewpoint_cam = Camera(
                FoVx=fovx,
                FoVy=fovy,
                image_width=batch["width"],
                image_height=batch["height"],
                world_view_transform=w2c,
                full_proj_transform=proj,
                camera_center=cam_p,
            )
            
            # Render
            with torch.amp.autocast('cuda', enabled=False):
                render_pkg = self.forward(
                    viewpoint_cam, self.background_tensor, **batch
                )
                
                renders.append(render_pkg["render"])
                viewspace_points.append(render_pkg["viewspace_points"])
                visibility_filters.append(render_pkg["visibility_filter"])
                radiis.append(render_pkg["radii"])
                
                if "normal" in render_pkg:
                    normals.append(render_pkg["normal"])
                if "pred_normal" in render_pkg and render_pkg["pred_normal"] is not None:
                    pred_normals.append(render_pkg["pred_normal"])
                if "depth" in render_pkg:
                    depths.append(render_pkg["depth"])
                if "mask" in render_pkg:
                    masks.append(render_pkg["mask"])
                if "lang" in render_pkg:
                    langs.append(render_pkg["lang"])
        
        # Stack outputs
        outputs = {
            "comp_rgb": torch.stack(renders, dim=0).permute(0, 2, 3, 1),
            "lang": torch.stack(langs, dim=0).permute(0, 2, 3, 1) if len(langs) > 0 else None,
            "viewspace_points": viewspace_points,
            "visibility_filter": visibility_filters,
            "radii": radiis,
        }
        
        if len(normals) > 0:
            outputs["comp_normal"] = torch.stack(normals, dim=0).permute(0, 2, 3, 1)
        if len(pred_normals) > 0:
            outputs["comp_pred_normal"] = torch.stack(pred_normals, dim=0).permute(0, 2, 3, 1)
        if len(depths) > 0:
            outputs["comp_depth"] = torch.stack(depths, dim=0).permute(0, 2, 3, 1)
        if len(masks) > 0:
            outputs["comp_mask"] = torch.stack(masks, dim=0).permute(0, 2, 3, 1)
        
        return outputs
