"""
3D Gaussian Splatting Geometry Model
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple, Optional, Dict, Any
from dataclasses import dataclass
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
import cv2
from PIL import Image
import logging

try:
    from diffusers import StableDiffusionInpaintPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logging.warning("diffusers not installed. Background inpainting will be disabled.")

# Constants
C0 = 0.28209479177387814  # SH constant


def RGB2SH(rgb):
    """Convert RGB to spherical harmonics"""
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    """Convert spherical harmonics to RGB"""
    return sh * C0 + 0.5


def inverse_sigmoid(x):
    """Inverse sigmoid function"""
    return torch.log(x / (1 - x))


def build_rotation(r):
    """
    Build rotation matrix from quaternion
    Args:
        r: [N, 4] quaternions
    Returns:
        R: [N, 3, 3] rotation matrices
    """
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )
    q = r / norm[:, None]
    
    R = torch.zeros((q.size(0), 3, 3), device=q.device)
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    """
    Build combined scaling and rotation matrix
    Args:
        s: [N, 3] scaling
        r: [N, 4] quaternion rotation
    Returns:
        L: [N, 3, 3] combined matrix
    """
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=s.device)
    R = build_rotation(r)
    
    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]
    
    L = R @ L
    return L


def strip_lowerdiag(L):
    """Extract lower diagonal elements"""
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device=L.device)
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    """Strip symmetric matrix"""
    return strip_lowerdiag(sym)


class BasicPointCloud(NamedTuple):
    """Basic point cloud structure"""
    points: np.array
    colors: np.array
    normals: np.array


@dataclass
class GaussianModelConfig:
    """Configuration for Gaussian model"""
    sh_degree: int = 3
    lang_feature_dim: int = 3  # CUDA rasterizer requires 3Ds 3D
    
    # Learning rates
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    lang_lr: float = 0.005
    
    # Optimizer params
    lang_beta_1: float = 0.9
    lang_beta_2: float = 0.999
    
    # Densification
    percent_dense: float = 0.01
    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 15000
    densify_grad_threshold: float = 0.0002
    
    # Pruning
    prune_from_iter: int = 500
    prune_until_iter: int = 15000
    prune_interval: int = 100
    min_opacity: float = 0.005
    
    # Initialization
    opacity_init: float = 0.1
    max_scaling: float = 100.0
    color_clip: float = 2.0


class GaussianBaseModel(nn.Module):
    """
    3D Gaussian Splatting Base Model
    Full implementation based on 3DitScene
    """
    
    def __init__(self, cfg: GaussianModelConfig):
        super().__init__()
        self.cfg = cfg
        
        self.active_sh_degree = 0
        self.max_sh_degree = cfg.sh_degree
        
        # Gaussian parameters
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._language_feature = torch.empty(0)
        
        # Training state
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.lang_optimizer = None
        self.spatial_lr_scale = 1.0
        
        # Setup activation functions
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = self.build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        
        # Background inpainting support
        self.inpainting_pipe = None
        self.inpainting_enabled = False
        
    
    def build_covariance_from_scaling_rotation(self, scaling, scaling_modifier, rotation):
        """Build 3D covariance matrix"""
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling).clamp(0, self.cfg.max_scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc.clamp(-self.cfg.color_clip, self.cfg.color_clip)
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_language_feature(self):
        """Get language features (normalized)"""
        return self._language_feature
    
    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def oneupSHdegree(self):
        """Increase SH degree"""
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
    
    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        """
        Initialize Gaussians from point cloud
        
        Args:
            pcd: Point cloud with points, colors, normals
            spatial_lr_scale: Spatial learning rate scale
        """
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0
        
        #print(f"[INFO] Number of points at initialization: {fused_point_cloud.shape[0]}")
        
        # Compute initial scales from nearest neighbors
        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        
        # Initialize rotations
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        
        # Initialize opacities
        opacities = inverse_sigmoid(self.cfg.opacity_init * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        
        # Initialize language features with random noise to break symmetry
        language_features = torch.randn((fused_point_cloud.shape[0], self.cfg.lang_feature_dim), device="cuda") * 0.01
        
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._language_feature = nn.Parameter(language_features.requires_grad_(True))
        
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    
    def training_setup(self):
        """Setup optimizers for training"""
        self.percent_dense = self.cfg.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        
        # Freeze language features during RGB phase
        self._xyz.requires_grad_(True)
        self._features_dc.requires_grad_(True)
        self._features_rest.requires_grad_(True)
        self._scaling.requires_grad_(True)
        self._rotation.requires_grad_(True)
        self._opacity.requires_grad_(True)
        self._language_feature.requires_grad_(False)
        
        l = [
            {'params': [self._xyz], 'lr': self.cfg.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': self.cfg.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': self.cfg.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': self.cfg.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': self.cfg.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': self.cfg.rotation_lr, "name": "rotation"},
        ]
        
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        #print("[INFO] Training setup complete")
    
    def lang_training_setup(self):
        """Setup optimizer for language feature training"""
        # Freeze geometry parameters
        self._xyz.requires_grad_(False)
        self._features_dc.requires_grad_(False)
        self._features_rest.requires_grad_(False)
        self._scaling.requires_grad_(False)
        self._rotation.requires_grad_(False)
        self._opacity.requires_grad_(False)
        
        # Enable language feature training
        self._language_feature.requires_grad_(True)
        
        l = [{'params': [self._language_feature], 'lr': self.cfg.lang_lr, "name": "language_feature"}]
        
        self.lang_optimizer = torch.optim.Adam(
            l, lr=0.0, eps=1e-15,
            betas=(self.cfg.lang_beta_1, self.cfg.lang_beta_2)
        )
        
        #print("[INFO] Language feature training setup complete")

    def setup_language_optimizer(self):
        """Setup optimizer for language feature training (Joint Training Mode)"""
        # Ensure language feature requires grad
        self._language_feature.requires_grad_(True)
        
        l = [{'params': [self._language_feature], 'lr': self.cfg.lang_lr, "name": "language_feature"}]
        
        self.lang_optimizer = torch.optim.Adam(
            l, lr=0.0, eps=1e-15,
            betas=(self.cfg.lang_beta_1, self.cfg.lang_beta_2)
        )
        
        print("[INFO] Language optimizer initialized for joint training")
    
    def update_learning_rate(self, iteration):
        """Update learning rate with exponential decay"""
        # Skip if optimizer not initialized (inference mode)
        if self.optimizer is None:
            return None
            
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.get_expon_lr_func(
                    lr_init=self.cfg.position_lr_init * self.spatial_lr_scale,
                    lr_final=self.cfg.position_lr_final * self.spatial_lr_scale,
                    lr_delay_mult=self.cfg.position_lr_delay_mult,
                    max_steps=self.cfg.position_lr_max_steps
                )(iteration)
                param_group['lr'] = lr
                return lr
    
    def get_expon_lr_func(self, lr_init, lr_final, lr_delay_mult=1.0, max_steps=1000000):
        """Get exponential learning rate function"""
        def helper(step):
            if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
                return 0.0
            if lr_delay_mult < 1:
                delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(0.5 * np.pi * np.clip(step / max_steps, 0, 1))
            else:
                delay_rate = 1.0
            t = np.clip(step / max_steps, 0, 1)
            log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
            return delay_rate * log_lerp
        return helper
    
    def reset_opacity(self):
        """Reset opacity for training stability"""
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
    
    def replace_tensor_to_optimizer(self, tensor, name):
        """Replace tensor in optimizer"""
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def _prune_optimizer(self, mask):
        """Prune optimizer parameters"""
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def prune_points(self, mask):
        """Prune Gaussians based on mask"""
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        self._language_feature = nn.Parameter(self._language_feature[valid_points_mask].requires_grad_(True))
        
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
    
    def cat_tensors_to_optimizer(self, tensors_dict):
        """Concatenate new tensors to optimizer"""
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        
        return optimizable_tensors
    
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        """Add new Gaussians after densification"""
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation
        }
        
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        # Extend language features with zeros
        new_lang_features = torch.zeros((new_xyz.shape[0], self._language_feature.shape[1]), device="cuda")
        self._language_feature = nn.Parameter(torch.cat([self._language_feature, new_lang_features], dim=0).requires_grad_(True))
        
        # Update language optimizer if it exists
        if hasattr(self, 'lang_optimizer') and self.lang_optimizer is not None:
            for param_group in self.lang_optimizer.param_groups:
                if param_group["name"] == "language_feature":
                    old_param = param_group["params"][0]
                    stored_state = self.lang_optimizer.state.get(old_param, None)
                    if stored_state is not None:
                        # Extend momentum buffers with zeros for the new Gaussians
                        stored_state["exp_avg"] = torch.cat([
                            stored_state["exp_avg"],
                            torch.zeros_like(new_lang_features),
                        ], dim=0)
                        stored_state["exp_avg_sq"] = torch.cat([
                            stored_state["exp_avg_sq"],
                            torch.zeros_like(new_lang_features),
                        ], dim=0)
                        del self.lang_optimizer.state[old_param]
                        self.lang_optimizer.state[self._language_feature] = stored_state
                    param_group["params"] = [self._language_feature]
        
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        """Densify and split large Gaussians"""
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        )
        
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
    
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """Clone small Gaussians with large gradients"""
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent
        )
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
    
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        """Adaptive density control"""
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        
        torch.cuda.empty_cache()
    
    def random_rotate(self, rotate_scale: float, apply: bool = True):
        """
        Apply random rotation augmentation to Gaussians
        
        Args:
            rotate_scale: Maximum rotation angle in degrees
            apply: Whether to actually apply the rotation
        """
        if not apply:
            return
        
        # Random rotation angle around Y-axis (yaw)
        angle = (random.random() * 2 - 1) * rotate_scale  # [-rotate_scale, +rotate_scale]
        angle_rad = angle * np.pi / 180.0
        
        # Build rotation matrix around Y-axis
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Rotation matrix (3x3) for Y-axis rotation
        R = torch.tensor([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ], dtype=torch.float32, device=self._xyz.device)
        
        # Clone and rotate positions around scene center
        # CRITICAL: Use clone() and create new Parameter to avoid in-place modification
        prev_xyz = self._xyz.clone()
        center = prev_xyz.mean(dim=0, keepdim=True)
        new_xyz = (prev_xyz - center) @ R.T + center
        self._xyz = nn.Parameter(new_xyz)
        
        # Update optimizer to track the new parameter
        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                if param_group["name"] == "xyz":
                    # Transfer optimizer state if it exists
                    old_param = param_group["params"][0]
                    stored_state = self.optimizer.state.get(old_param, None)
                    if stored_state is not None:
                        del self.optimizer.state[old_param]
                        self.optimizer.state[self._xyz] = stored_state
                    param_group["params"][0] = self._xyz
                    break
    
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        """Track gradient statistics"""
        if viewspace_point_tensor.grad is None:
            return
            
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
    
    def save_ply(self, path):
        """Save Gaussians to PLY file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        lang_feat = self._language_feature.detach().cpu().numpy()
        
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, lang_feat), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
        #print(f"[INFO] Saved PLY to: {path}")
    
    def load_ply(self, path):
        """Load Gaussians from PLY file"""
        plydata = PlyData.read(path)
        
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        # Load language features
        lang_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("lang_")]
        if len(lang_names) > 0:
            lang_names = sorted(lang_names, key=lambda x: int(x.split('_')[-1]))
            lang_feat = np.zeros((xyz.shape[0], len(lang_names)))
            for idx, attr_name in enumerate(lang_names):
                lang_feat[:, idx] = np.asarray(plydata.elements[0][attr_name])
        else:
            lang_feat = np.zeros((xyz.shape[0], self.cfg.lang_feature_dim))
        
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._language_feature = nn.Parameter(torch.tensor(lang_feat, dtype=torch.float, device="cuda").requires_grad_(True))
        
        self.active_sh_degree = self.max_sh_degree
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        #print(f"[INFO] Loaded {xyz.shape[0]} Gaussians from: {path}")
    
    def construct_list_of_attributes(self):
        """Construct list of attributes for PLY file"""
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._language_feature.shape[1]):
            l.append('lang_{}'.format(i))
        return l
