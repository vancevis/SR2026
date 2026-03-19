"""
Camera utilities for rendering
"""

import math
import torch
from typing import NamedTuple, Tuple


class Camera(NamedTuple):
    """Camera parameters for rendering"""
    FoVx: float
    FoVy: float
    image_width: int
    image_height: int
    world_view_transform: torch.Tensor
    full_proj_transform: torch.Tensor
    camera_center: torch.Tensor


def get_cam_info_gaussian(
    c2w: torch.Tensor,
    fovx: float,
    fovy: float,
    znear: float = 0.1,
    zfar: float = 100.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get camera matrices from camera-to-world transform
    
    Args:
        c2w: [4, 4] camera-to-world transform
        fovx: Field of view in x (radians)
        fovy: Field of view in y (radians)
        znear: Near plane
        zfar: Far plane
    
    Returns:
        w2c: World-to-camera matrix [4, 4]
        proj: Full projection matrix [4, 4]
        cam_pos: Camera position [3]
        cam_proj: Camera projection matrix [4, 4]
    """
    # World-to-camera (analytical inverse for rigid-body transforms)
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    w2c = torch.eye(4, device=c2w.device, dtype=c2w.dtype)
    w2c[:3, :3] = R.T
    w2c[:3, 3] = -R.T @ t
    
    # Projection matrix
    tanHalfFovY = math.tan(fovy / 2)
    tanHalfFovX = math.tan(fovx / 2)
    
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right
    
    P = torch.zeros(4, 4, device=c2w.device, dtype=c2w.dtype)
    z_sign = 1.0
    
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    
    # Full projection
    full_proj = w2c.T @ P
    
    # Camera position
    cam_pos = c2w[:3, 3]
    
    return w2c.T.contiguous(), full_proj.T.contiguous(), cam_pos, P
