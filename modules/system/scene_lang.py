"""
Scene Language System
Complete training system with language feature distillation
"""

import os
import math
import random
import logging
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm, trange
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from ..geometry.gaussian_base import GaussianBaseModel, GaussianModelConfig, BasicPointCloud
from ..renderer import DiffGaussianRenderer, GaussianBatchRenderer, DiffGaussianBatchRenderer
from ..renderer.camera import Camera, get_cam_info_gaussian
from ..utils.sam_clip import SamClip, SamClipConfig
from ..utils.ae import Autoencoder, AutoencoderDataset
from ..utils.loss import l2_loss, cos_loss, tv_loss, ssim_loss

logger = logging.getLogger(__name__)

@dataclass
class SceneLangConfig:
    """Configuration for Scene Language System"""
    # Geometry
    geometry: GaussianModelConfig = field(default_factory=GaussianModelConfig)
    
    # Renderer
    invert_bg_prob: float = 1.0
    back_ground_color: tuple = (1.0, 1.0, 1.0)
    
    # Language distillation
    distill_lang_freq: int = 100
    distill_lang_epoch: int = 100
    distill_interval: int = 2
    
    # SAM+CLIP
    sam_clip: SamClipConfig = field(default_factory=SamClipConfig)
    
    # Autoencoder
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64, 32, 3])  # 3D latent (CUDA requirement)
    decoder_hidden_dims: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 512])  # Back to 512D
    ae_epoch: int = 100
    sam_clip_ae_lr: float = 3e-4
    
    # Training
    densify: bool = True
    xyz_noise_ratio: float = 0.0
    
    # Loss weights
    lambda_rgb: float = 1.0
    lambda_dssim: float = 0.2
    lambda_sds: float = 1.0
    lambda_ref: float = 0.0
    lambda_scaling: float = 0.0
    
    # Prompt
    prompt: str = "a scene"
    empty_prompt: str = "empty"
    side_prompt: str = "empty"
    
    # Outpainting
    outpaint_step: int = 300
    crop_with_lang: bool = True
    rotate_aug_scale: int = 15


class SceneLangSystem(nn.Module):
    """
    Complete scene language training system
    Integrates RGB training and language feature distillation
    """
    
    def __init__(
        self,
        cfg: SceneLangConfig = None,
        device: str = "cuda"
    ):
        super().__init__()
        self.cfg = cfg or SceneLangConfig()
        self.device = device
        
        # Initialize geometry
        self.geometry = GaussianBaseModel(self.cfg.geometry).to(device)
        self.geometry.prompt = self.cfg.prompt
        self.geometry.empty_prompt = self.cfg.empty_prompt
        self.geometry.side_prompt = self.cfg.side_prompt
        
        # Initialize renderer with batch rendering capability
        self.renderer = DiffGaussianBatchRenderer(
            self.geometry,
            cfg=type('Config', (), {
                'debug': False,
                'invert_bg_prob': self.cfg.invert_bg_prob,
                'back_ground_color': self.cfg.back_ground_color
            })()
        )
        
        # Initialize SAM+CLIP
        self.sam_clip = SamClip(self.cfg.sam_clip)
        
        # Initialize Autoencoder (512D CLIP -> 32D latent)
        self.sam_clip_ae = Autoencoder(
            input_dim=512,
            latent_dim=self.cfg.geometry.lang_feature_dim,  # Match geometry feature dim
            encoder_hidden_dims=self.cfg.encoder_hidden_dims,
            decoder_hidden_dims=self.cfg.decoder_hidden_dims
        ).to(device)
        
        # Training state
        self.global_step = 0
        self.outpaint_view = {}
        self.outpaint_mask = {}
        
        # Semantic targets cache
        self.semantic_targets = {}
        
        # Optimizer (set by training_setup)
        self.optimizer = None
        
    
    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float = 1.0):
        """Initialize geometry from point cloud"""
        self.geometry.create_from_pcd(pcd, spatial_lr_scale)
    
    def training_setup(self):
        """Setup training"""
        self.geometry.training_setup()
        self.optimizer = self.geometry.optimizer
        self.renderer.training = True
    
    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass
        
        Args:
            batch: Dict with 'c2w', 'fovy', 'width', 'height'
        
        Returns:
            Rendered outputs
        """
        self.geometry.update_learning_rate(self.global_step)
        outputs = self.renderer.batch_forward(batch)
        return outputs
    
    def training_step(
        self,
        batch: Dict[str, Any],
        guidance_out: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Single training step
        
        Args:
            batch: Training batch
            guidance_out: Optional guidance output (e.g., from diffusion model)
        
        Returns:
            Dict with loss and metrics
        """
        # Apply noise
        self.geometry.noise_ratio = self.cfg.xyz_noise_ratio
        
        # Outpainting at specific step
        if self.global_step == self.cfg.outpaint_step:
            self.outpaint(batch)
        
        # Random rotation augmentation after language distillation
        apply_rotate = False
        if self.global_step > self.cfg.distill_lang_freq:
            apply_rotate = random.random() < 0.5
            if apply_rotate:
                self.geometry.random_rotate(self.cfg.rotate_aug_scale, apply_rotate)
        
        # Forward pass
        out = self(batch)
        
        visibility_filter = out["visibility_filter"]
        radii = out["radii"]
        guidance_inp = out["comp_rgb"]
        viewspace_point_tensor = out["viewspace_points"]
        
        # Retain gradients for densification
        if isinstance(viewspace_point_tensor, list):
            for vp in viewspace_point_tensor:
                vp.retain_grad()
        else:
            viewspace_point_tensor.retain_grad()
        
        # Compute loss
        loss = 0.0
        loss_rgb = 0.0
        loss_lang = 0.0
        
        # Language Feature Loss (Joint Training)
        if hasattr(self.geometry, 'lang_optimizer') and self.geometry.lang_optimizer is not None:
            if 'index' in batch:
                indices = batch['index']
                rendered_langs = out.get('lang', None)
                
                if rendered_langs is not None:
                    targets = []
                    preds = []
                    
                    for i, idx in enumerate(indices):
                        idx_val = idx.item()
                        if idx_val in self.semantic_targets:
                            target = self.semantic_targets[idx_val].to(self.device)
                            pred = rendered_langs[i]
                            
                            if pred.shape != target.shape:
                                target = F.interpolate(
                                    target.permute(2, 0, 1).unsqueeze(0),
                                    size=(pred.shape[0], pred.shape[1]),
                                    mode='bilinear',
                                    align_corners=False
                                ).squeeze(0).permute(1, 2, 0)
                            
                            targets.append(target)
                            preds.append(pred)
                    
                    if len(targets) > 0:
                        target_tensor = torch.stack(targets)
                        pred_tensor = torch.stack(preds)
                        
                        l2 = l2_loss(target_tensor, pred_tensor)
                        cos = cos_loss(target_tensor.reshape(-1, 3), pred_tensor.reshape(-1, 3))
                        loss_lang = l2 + cos * 0.001
                        loss += loss_lang
        
        # RGB Reconstruction Loss
        if "image" in batch:
            gt_image = batch["image"].to(self.device)
            # [B, H, W, 3] -> [B, 3, H, W]
            pred_image = out["comp_rgb"].permute(0, 3, 1, 2)
            
            # Resize GT if necessary (though usually they match)
            if gt_image.shape[2:] != pred_image.shape[2:]:
                gt_image = F.interpolate(gt_image, size=pred_image.shape[2:], mode="bilinear", align_corners=False)
            
            l1 = F.l1_loss(pred_image, gt_image)
            ssim = ssim_loss(pred_image, gt_image)
            
            loss_rgb = (1.0 - self.cfg.lambda_dssim) * l1 + self.cfg.lambda_dssim * ssim
            loss += loss_rgb * self.cfg.lambda_rgb
            
        loss_sds = 0.0
        
        # SDS loss from guidance
        if guidance_out is not None:
            for name, value in guidance_out.items():
                if name.startswith("loss_"):
                    loss_sds += value * self.cfg.lambda_sds
        
        # Scaling regularization
        if self.cfg.lambda_scaling > 0.0:
            scaling_loss = self.geometry.get_scaling.mean()
            loss += scaling_loss * self.cfg.lambda_scaling
        
        loss = loss + loss_sds
        
        # Backward
        self.optimizer.zero_grad()
        if hasattr(self.geometry, 'lang_optimizer') and self.geometry.lang_optimizer is not None:
            self.geometry.lang_optimizer.zero_grad()
        
        if loss > 0:
            loss.backward()
        
        # Update densification stats
        if self.cfg.densify:
            for batch_idx, (vp, vf, r) in enumerate(zip(viewspace_point_tensor, visibility_filter, radii)):
                self.geometry.add_densification_stats(vp, vf)
        
        # Optimizer step for geometry
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        
        # CRITICAL: Step language optimizer if joint training is active
        if hasattr(self.geometry, 'lang_optimizer') and self.geometry.lang_optimizer is not None:
            self.geometry.lang_optimizer.step()
            self.geometry.lang_optimizer.zero_grad(set_to_none=True)
        
        self.global_step += 1
        
        return {
            "loss": loss,
            "loss_rgb": loss_rgb,
            "loss_lang": loss_lang,
            "loss_sds": loss_sds,
            "num_gaussians": self.geometry.get_xyz.shape[0]
        }
    
    def prepare_semantic_data(self, dataloader_or_batch: Any) -> None:
        """
        Prepare semantic data for joint training:
        1. Extract SAM+CLIP features
        2. Train Autoencoder
        3. Cache encoded features as targets
        4. Initialize language optimizer
        
        Args:
            dataloader_or_batch: DataLoader or batch dict
        """
        
        total_embed = []
        total_feat = []
        total_flag = []

        # BUG-1 fix: initialize cache for training batches
        # This is used by _build_finetune_batch during post-edit semantic update
        self.cached_batches = []
        
        # Extract SAM+CLIP features from rendered views
        
        # Debug: Print type of input
        # print(f"[DEBUG] distill_language_feature input type: {type(dataloader_or_batch)}")
        
        # Handle both DataLoader and list of samples
        # Check if it is a single batch (dict-like)
        if hasattr(dataloader_or_batch, 'keys'):
            # Single batch
            dataset = [dataloader_or_batch]
            indices = [0]
        elif hasattr(dataloader_or_batch, '__len__'):
            dataset = dataloader_or_batch
            indices = range(0, len(dataset), self.cfg.distill_interval)
        else:
            # Single batch (fallback)
            dataset = [dataloader_or_batch]
            indices = [0]
        
        for idx in trange(len(indices), desc="Extracting features"):
            actual_idx = indices[idx] if isinstance(indices, range) else idx
            
            sample = dataset[actual_idx]
            
            # Ensure batch format
            for k in sample.keys():
                if not isinstance(sample[k], torch.Tensor):
                    continue
                if sample[k].dim() == 0 or (sample[k].dim() > 0 and sample[k].shape[0] != 1):
                    try:
                        sample[k] = sample[k].cuda()[None] if isinstance(sample[k], torch.Tensor) else sample[k]
                    except (RuntimeError, ValueError):
                        pass
            
            # Extract features from GT image if available, else render
            if 'image' in sample:
                # GT Image: [1, 3, H, W] float 0-1
                rgb = sample['image']
                if rgb.dtype == torch.float32:
                    rgb = (rgb * 255).clamp(0, 255).to(torch.uint8)
            else:
                # Render
                with torch.no_grad():
                    output = self(sample)
                
                rgb = output['comp_rgb']  # [1, H, W, 3]
                rgb = (rgb.permute(0, 3, 1, 2) * 255).clamp(0, 255).to(torch.uint8)
            
            try:
                embed, seg, mask = self.sam_clip(rgb)
                total_embed.append(embed.cpu())
                total_feat.append(seg.cpu())
                total_flag.append(actual_idx)

                # Cache batch for semantic finetune (store on CPU to save GPU memory)
                self.cached_batches.append({
                    k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
                    for k, v in sample.items()
                })
            except Exception as e:
                logger.warning(
                    "[prepare_semantic_data] Feature extraction failed for view %d: %s",
                    actual_idx, e,
                )
                continue
        
        if len(total_embed) == 0:
            return
        
        # Step 2: Train Autoencoder
        
        all_embeds = torch.cat(total_embed, 0).float().numpy()
        ae_dataset = AutoencoderDataset(all_embeds)
        ae_dataloader = DataLoader(ae_dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=False)
        
        optimizer = torch.optim.Adam(self.sam_clip_ae.parameters(), lr=self.cfg.sam_clip_ae_lr)
        
        self.sam_clip_ae.train()
        for epoch in tqdm(range(self.cfg.ae_epoch), desc="Training autoencoder"):
            for data in ae_dataloader:
                data = data.to(self.device)
                mid = self.sam_clip_ae.encode(data)
                _data = self.sam_clip_ae.decode(mid)
                
                l2loss = l2_loss(_data, data)
                cosloss = cos_loss(_data, data)
                loss = l2loss + cosloss * 0.001
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Step 3: Encode features
        
        self.sam_clip_ae.eval()
        mids = dict()
        
        # Get image dimensions
        if isinstance(dataset, list):
            H, W = dataset[0].get('height', 512), dataset[0].get('width', 512)
        else:
            H, W = 512, 512
        
        with torch.no_grad():
            # Background embedding: normalized CLIP feature for empty regions
            zero_tensor = torch.zeros([1, 512], dtype=torch.float32)
            zero_tensor = zero_tensor / (zero_tensor.norm(dim=-1, keepdim=True) + 1.0)  # Normalized to small value
            
            for idx, seg, embed in zip(total_flag, total_feat, total_embed):
                embeds = torch.cat([embed, zero_tensor], 0).float().to(self.device)
                embeds = self.sam_clip_ae.encode(embeds)
                
                mid = embeds[seg.long().to(self.device)].reshape(H, W, -1)
                mids[idx] = mid
        
        # Step 4: Cache targets and setup optimizer for joint training
        self.semantic_targets = mids
        
        # Validate feature dimensions
        if len(mids) > 0:
            sample_idx = list(mids.keys())[0]
            sample_shape = mids[sample_idx].shape
            expected_dim = self.cfg.geometry.lang_feature_dim
            actual_dim = sample_shape[-1]
            print(f"[INFO] Semantic targets prepared for {len(self.semantic_targets)} views")
            print(f"[INFO] Target feature shape: {sample_shape}, expected dim: {expected_dim}, actual dim: {actual_dim}")
            
            if actual_dim != expected_dim:
                raise RuntimeError(
                    f"Semantic target dimension mismatch! "
                    f"Expected {expected_dim}D features but got {actual_dim}D. "
                    f"This likely means:\n"
                    f"1. Old code is still running on server (not synced)\n"
                    f"2. Or autoencoder latent_dim config is wrong\n"
                    f"Please sync code and restart training from scratch."
                )
        
        # Initialize language optimizer
        if hasattr(self.geometry, 'setup_language_optimizer'):
            self.geometry.setup_language_optimizer()
        else:
            print("[WARNING] geometry.setup_language_optimizer not found, falling back to lang_training_setup")
            self.geometry.lang_training_setup()

    def local_semantic_finetune(
        self,
        affected_mask: torch.Tensor,
        num_steps: int = 30,
        lr: float = 0.005,
        neighborhood_radius: float = 0.1,
        equivariance_weight: float = 0.1,
    ) -> None:
        """
        Post-edit local semantic field update with edit-equivariance
        constraint (Direction A innovation).

        After an editing operation (translate, rotate, scale, delete), the
        semantic features in the affected region may become inconsistent.
        This method performs localized fine-tuning of language features to
        restore semantic coherence.

        Algorithm:
            1. Extend affected region: edited Gaussians + spatial neighbors.
            2. Initialize zero-feature Gaussians from nearest non-zero neighbors.
            3. Freeze all geometry parameters; optimize only language features.
            4. Run gradient descent with the cached training-view targets
               PLUS an edit-equivariance regularizer that enforces:
               - Edited Gaussians preserve their pre-edit semantic identity
                 (spatial coherence constraint).
               - Non-edited neighbors maintain smooth feature gradients
                 towards the edited boundary (boundary smoothness).

        Args:
            affected_mask: Boolean mask of edited Gaussians, shape [N].
            num_steps: Number of fine-tuning gradient steps.
            lr: Learning rate for the temporary optimizer.
            neighborhood_radius: Spatial radius for extending the affected region.
            equivariance_weight: Weight for the edit-equivariance loss term.
        """
        # Guard: check if semantic targets are available
        if not self.semantic_targets:
            logger.warning(
                "[SemanticFinetune] No cached semantic targets, skipping finetune"
            )
            return

        num_affected = affected_mask.sum().item()
        if num_affected == 0:
            logger.warning("[SemanticFinetune] Empty affected mask, skipping")
            return

        # ---------------------------------------------------------------
        # Step 1: Extend affected region — density-adaptive radius (Dir E)
        # ---------------------------------------------------------------
        # Instead of a fixed radius, compute a local density-adaptive
        # radius for each affected Gaussian based on its k-NN distance.
        # This ensures consistent spatial coverage: dense regions get a
        # smaller radius, sparse regions get a larger one.
        xyz = self.geometry.get_xyz.detach()  # [N, 3]
        affected_centers = xyz[affected_mask]  # [M, 3]

        # Compute adaptive radius from affected Gaussians' local density
        knn_k = min(16, affected_centers.shape[0] - 1)
        if knn_k >= 1:
            aff_dists = torch.cdist(affected_centers, affected_centers)
            aff_dists.fill_diagonal_(float("inf"))
            knn_dists, _ = aff_dists.topk(knn_k, dim=1, largest=False)
            # Per-Gaussian adaptive radius: scale × mean k-NN distance
            per_gaussian_radius = knn_dists.mean(dim=1) * 2.0  # [M]
            # Clamp to [neighborhood_radius * 0.2, neighborhood_radius * 5]
            # to avoid degenerate cases
            r_min = neighborhood_radius * 0.2
            r_max = neighborhood_radius * 5.0
            per_gaussian_radius = per_gaussian_radius.clamp(r_min, r_max)
            adaptive_radius = per_gaussian_radius.median().item()
        else:
            adaptive_radius = neighborhood_radius

        chunk_size = 4096
        extended_mask = affected_mask.clone()

        for start_idx in range(0, xyz.shape[0], chunk_size):
            end_idx = min(start_idx + chunk_size, xyz.shape[0])
            chunk_xyz = xyz[start_idx:end_idx]  # [C, 3]
            dists = torch.cdist(chunk_xyz, affected_centers)  # [C, M]
            if knn_k >= 1:
                # Per-Gaussian threshold: each column j uses radius[j]
                within_radius = dists < per_gaussian_radius.unsqueeze(0)
                extended_mask[start_idx:end_idx] |= within_radius.any(dim=1)
            else:
                min_dists = dists.min(dim=1).values
                extended_mask[start_idx:end_idx] |= (min_dists < adaptive_radius)

        num_extended = extended_mask.sum().item()
        logger.info(
            "[SemanticFinetune] Affected: %d → Extended: %d "
            "(adaptive_radius median=%.4f, range=[%.4f, %.4f])",
            num_affected,
            num_extended,
            adaptive_radius,
            per_gaussian_radius.min().item() if knn_k >= 1 else adaptive_radius,
            per_gaussian_radius.max().item() if knn_k >= 1 else adaptive_radius,
        )

        # ---------------------------------------------------------------
        # Step 2: Initialize zero-feature Gaussians from nearest neighbors
        # ---------------------------------------------------------------
        lang_features = self.geometry._language_feature  # nn.Parameter [N, d]
        feat_norms = lang_features.data.norm(dim=-1)  # [N]
        zero_mask = (feat_norms < 1e-6) & extended_mask
        num_zero = zero_mask.sum().item()

        if num_zero > 0:
            nonzero_mask = feat_norms >= 1e-6
            if nonzero_mask.sum() > 0:
                nonzero_xyz = xyz[nonzero_mask]
                nonzero_feat = lang_features.data[nonzero_mask]
                zero_xyz = xyz[zero_mask]

                nn_dists = torch.cdist(zero_xyz, nonzero_xyz)  # [Z, K]
                nn_idx = nn_dists.argmin(dim=1)  # [Z]

                with torch.no_grad():
                    lang_features.data[zero_mask] = nonzero_feat[nn_idx].clone()

                logger.info(
                    "[SemanticFinetune] Initialized %d zero-feature Gaussians "
                    "from nearest neighbors",
                    num_zero,
                )

        # ---------------------------------------------------------------
        # Step 2b: Cache pre-finetune features for equivariance constraint
        # ---------------------------------------------------------------
        # The equivariance prior: edited Gaussians should preserve their
        # semantic identity (feature direction) while adapting magnitude.
        # This prevents catastrophic semantic drift during fine-tuning.
        pre_finetune_feat = lang_features.data[affected_mask].clone().detach()
        pre_finetune_norms = pre_finetune_feat.norm(dim=-1, keepdim=True)
        pre_finetune_dirs = pre_finetune_feat / (pre_finetune_norms + 1e-9)

        # Boundary smoothness: identify border Gaussians (extended but not affected)
        border_mask = extended_mask & ~affected_mask
        border_indices = torch.where(border_mask)[0]
        border_anchor_feat = lang_features.data[border_mask].clone().detach()

        # ---------------------------------------------------------------
        # Step 3: Localized gradient descent with equivariance constraint
        # ---------------------------------------------------------------
        lang_features.requires_grad_(True)
        temp_optimizer = torch.optim.Adam([lang_features], lr=lr)

        freeze_mask = (~extended_mask).unsqueeze(-1)

        cached_view_indices = list(self.semantic_targets.keys())
        if not cached_view_indices:
            logger.warning("[SemanticFinetune] No cached views for finetune")
            return

        # Uniform view sampling across cached views
        view_weights = None

        feature_dim = lang_features.shape[-1]

        # Direction F: EMA-based convergence detection for early stopping
        ema_loss = None
        ema_beta = 0.9
        converge_eps = 1e-4
        converge_patience = 5
        converge_count = 0
        actual_steps = 0

        for step in range(num_steps):
            if view_weights is not None:
                vi_local = np.random.choice(
                    len(cached_view_indices), p=view_weights
                )
                view_idx = cached_view_indices[vi_local]
            else:
                view_idx = cached_view_indices[
                    torch.randint(len(cached_view_indices), (1,)).item()
                ]

            target = self.semantic_targets[view_idx].to(self.device)  # [H, W, d]

            batch = self._build_finetune_batch(view_idx)
            if batch is None:
                continue

            out = self(batch)
            rendered_lang = out.get("lang")
            if rendered_lang is None:
                continue

            pred = rendered_lang[0]  # [H, W, d]

            if pred.shape[:2] != target.shape[:2]:
                target = F.interpolate(
                    target.permute(2, 0, 1).unsqueeze(0),
                    size=(pred.shape[0], pred.shape[1]),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).permute(1, 2, 0)

            # --- Reconstruction loss (L2 + cosine) ---
            # NOTE: cosine weight must match training_step (0.001) to stay
            # on the same loss manifold. Using a higher weight here would
            # push features off the training distribution.
            loss_l2 = l2_loss(pred, target)
            loss_cos = cos_loss(
                pred.reshape(-1, feature_dim),
                target.reshape(-1, feature_dim),
            )
            loss_recon = loss_l2 + loss_cos * 0.001

            # --- Edit-equivariance loss (Direction A innovation) ---
            # (a) Semantic identity preservation: cosine similarity between
            #     current and pre-edit feature directions for affected Gaussians
            loss_equivar = torch.tensor(0.0, device=self.device)
            if num_affected > 0 and equivariance_weight > 0:
                current_affected = lang_features[affected_mask]
                current_dirs = current_affected / (
                    current_affected.norm(dim=-1, keepdim=True) + 1e-9
                )
                # 1 - cos_sim: penalize direction change
                identity_loss = (
                    1.0 - F.cosine_similarity(current_dirs, pre_finetune_dirs, dim=-1)
                ).mean()

                # (b) Boundary smoothness: border Gaussians should stay
                #     close to their anchor values (hard stability constraint)
                border_loss = torch.tensor(0.0, device=self.device)
                if border_indices.numel() > 0:
                    current_border = lang_features[border_mask]
                    border_loss = (
                        (current_border - border_anchor_feat) ** 2
                    ).mean()

                loss_equivar = identity_loss * 0.7 + border_loss * 0.3

            loss = loss_recon + equivariance_weight * loss_equivar

            temp_optimizer.zero_grad()
            loss.backward()

            # Zero out gradients for Gaussians outside the extended region
            with torch.no_grad():
                if lang_features.grad is not None:
                    lang_features.grad[freeze_mask.expand_as(lang_features.grad)] = 0.0

            temp_optimizer.step()
            actual_steps += 1

            # NOTE: Do NOT apply L2 normalization here.
            # The renderer (diff_gaussian_rasterizer.py) already normalizes
            # language features on-the-fly before rasterization. Modifying
            # .data directly would also pollute Adam's momentum/variance states.

            # --- Direction F: Early stopping via EMA convergence ---
            loss_val = loss.item()
            if ema_loss is None:
                ema_loss = loss_val
            else:
                ema_loss = ema_beta * ema_loss + (1 - ema_beta) * loss_val
                relative_change = abs(loss_val - ema_loss) / (ema_loss + 1e-9)
                if relative_change < converge_eps:
                    converge_count += 1
                else:
                    converge_count = 0
                if converge_count >= converge_patience and step >= 10:
                    logger.info(
                        "[SemanticFinetune] Early stop: converged at step "
                        "%d/%d (ema_loss=%.6f, rel_change=%.2e)",
                        step + 1, num_steps, ema_loss, relative_change,
                    )
                    break

        logger.info(
            "[SemanticFinetune] Completed %d/%d finetune steps on %d Gaussians "
            "(equivariance_weight=%.3f, final_ema_loss=%.6f)",
            actual_steps,
            num_steps,
            num_extended,
            equivariance_weight,
            ema_loss if ema_loss is not None else 0.0,
        )

    # ------------------------------------------------------------------
    # Direction B: Semantic Target Warping
    # ------------------------------------------------------------------

    def warp_semantic_targets(
        self,
        pre_edit_xyz: torch.Tensor,
        post_edit_xyz: torch.Tensor,
    ) -> int:
        """
        Warp cached semantic targets to account for known geometric edits.

        When Gaussians are moved (translate/rotate/scale), the 2D semantic
        target maps become stale — they depict the object at its OLD position.
        This method warps the targets analytically using the known 3D
        correspondences (pre → post positions) projected into each view.

        Algorithm (per view):
            1. Project pre-edit and post-edit Gaussian positions to 2D.
            2. Estimate a 2D affine transform from the correspondences
               (handles translation, rotation, and scale in image space).
            3. Build an inverse-warp sampling grid and apply F.grid_sample.
            4. Fill vacated (dis-occluded) pixels with local background.

        Args:
            pre_edit_xyz: Pre-edit 3D positions of affected Gaussians [M, 3].
            post_edit_xyz: Post-edit 3D positions of affected Gaussians [M, 3].

        Returns:
            Number of target views successfully warped.
        """
        if not self.semantic_targets:
            return 0

        num_warped = 0

        for view_idx in list(self.semantic_targets.keys()):
            target = self.semantic_targets[view_idx]  # [H, W, D]
            batch = self._build_finetune_batch(view_idx)
            if batch is None:
                continue

            H, W = target.shape[:2]

            # Project pre/post positions to 2D
            uv_pre, valid_pre = self._project_to_2d(
                pre_edit_xyz, batch, H, W
            )
            uv_post, valid_post = self._project_to_2d(
                post_edit_xyz, batch, H, W
            )

            valid = valid_pre & valid_post
            K = valid.sum().item()
            if K < 3:
                continue

            uv_pre_v = uv_pre[valid]    # [K, 2]
            uv_post_v = uv_post[valid]  # [K, 2]

            # Build 2D masks for old / new object regions
            splat_r = max(3, min(H, W) // 60)
            mask_old = self._splat_to_mask(uv_pre_v, H, W, splat_r)
            mask_new = self._splat_to_mask(uv_post_v, H, W, splat_r)

            # Estimate 2D affine: uv_post ≈ [uv_pre, 1] @ A_fwd
            src_h = torch.cat(
                [uv_pre_v, torch.ones(K, 1, device=self.device)], dim=1
            )  # [K, 3]
            # Inverse affine: given new coords → old coords
            dst_h = torch.cat(
                [uv_post_v, torch.ones(K, 1, device=self.device)], dim=1
            )  # [K, 3]
            A_inv = torch.linalg.lstsq(dst_h, uv_pre_v).solution  # [3, 2]

            # Build sampling grid (default = identity)
            grid_v, grid_u = torch.meshgrid(
                torch.arange(H, device=self.device, dtype=torch.float32),
                torch.arange(W, device=self.device, dtype=torch.float32),
                indexing="ij",
            )
            sample_u = grid_u.clone()
            sample_v = grid_v.clone()

            # For new object region: sample from old position (inverse warp)
            new_pixels = mask_new.nonzero(as_tuple=False)  # [P, 2] (row, col)
            if new_pixels.shape[0] > 0:
                new_u = new_pixels[:, 1].float()
                new_v = new_pixels[:, 0].float()
                coords_h = torch.stack(
                    [new_u, new_v, torch.ones_like(new_u)], dim=1
                )  # [P, 3]
                src_coords = coords_h @ A_inv  # [P, 2]
                sample_u[new_pixels[:, 0], new_pixels[:, 1]] = src_coords[:, 0]
                sample_v[new_pixels[:, 0], new_pixels[:, 1]] = src_coords[:, 1]

            # Normalize to [-1, 1] for grid_sample
            norm_u = 2.0 * sample_u / max(W - 1, 1) - 1.0
            norm_v = 2.0 * sample_v / max(H - 1, 1) - 1.0
            grid = torch.stack([norm_u, norm_v], dim=-1).unsqueeze(0)

            # Warp target
            target_bchw = target.permute(2, 0, 1).unsqueeze(0).float()
            warped = F.grid_sample(
                target_bchw, grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            warped = warped.squeeze(0).permute(1, 2, 0)  # [H, W, D]

            # Fill vacated region (old object, not covered by new) with bg
            vacated = mask_old & ~mask_new
            if vacated.sum() > 0:
                bg_region = ~mask_old & ~mask_new
                if bg_region.sum() > 0:
                    bg_mean = target[bg_region].mean(dim=0)
                    warped[vacated] = bg_mean

            self.semantic_targets[view_idx] = warped.detach()
            num_warped += 1

        if num_warped > 0:
            logger.info(
                "[SemanticTargetWarp] Warped %d/%d target views",
                num_warped,
                len(self.semantic_targets),
            )
        return num_warped

    def _project_to_2d(
        self,
        xyz: torch.Tensor,
        batch: Dict[str, Any],
        H: int,
        W: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project 3D points to 2D pixel coordinates using camera from batch.

        Args:
            xyz: 3D points [M, 3].
            batch: Camera batch dict.
            H: Image height.
            W: Image width.

        Returns:
            uv: [M, 2] float tensor of (u, v) pixel coordinates.
            valid: [M] boolean mask (in front of camera AND within image).
        """
        M = xyz.shape[0]
        uv = torch.zeros(M, 2, device=self.device)
        valid = torch.zeros(M, dtype=torch.bool, device=self.device)

        c2w = batch.get("c2w")
        if c2w is None:
            return uv, valid

        c2w_mat = c2w[0] if c2w.dim() == 3 else c2w
        w2c = torch.inverse(c2w_mat)

        fovx = batch.get("fovx", batch.get("fov", 0.8))
        fovy = batch.get("fovy", fovx)
        if isinstance(fovx, torch.Tensor):
            fovx = fovx.item()
        if isinstance(fovy, torch.Tensor):
            fovy = fovy.item()

        fx = W / (2.0 * math.tan(fovx / 2.0))
        fy = H / (2.0 * math.tan(fovy / 2.0))
        cx, cy = W / 2.0, H / 2.0

        pts_h = torch.cat([
            xyz.to(self.device),
            torch.ones(M, 1, device=self.device),
        ], dim=1)
        pts_cam = (w2c @ pts_h.T).T[:, :3]

        in_front = pts_cam[:, 2] > 0.01
        if in_front.sum() > 0:
            u = fx * pts_cam[:, 0] / pts_cam[:, 2] + cx
            v = fy * pts_cam[:, 1] / pts_cam[:, 2] + cy
            uv[:, 0] = u
            uv[:, 1] = v
            valid = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)

        return uv, valid

    @staticmethod
    def _splat_to_mask(
        uv: torch.Tensor,
        H: int,
        W: int,
        radius: int,
    ) -> torch.Tensor:
        """
        Create boolean mask by splatting 2D points with a given radius.

        Uses conv2d-based morphological dilation for vectorized efficiency.

        Args:
            uv: [K, 2] float tensor of (u, v) pixel coordinates.
            H: Image height.
            W: Image width.
            radius: Dilation radius in pixels.

        Returns:
            Boolean mask [H, W].
        """
        device = uv.device
        mask = torch.zeros(H, W, dtype=torch.bool, device=device)

        u = uv[:, 0].long().clamp(0, W - 1)
        v = uv[:, 1].long().clamp(0, H - 1)
        mask[v, u] = True

        # Morphological dilation via convolution
        mask_f = mask.float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        kernel = torch.ones(
            1, 1, 2 * radius + 1, 2 * radius + 1, device=device
        )
        dilated = F.conv2d(mask_f, kernel, padding=radius)
        return dilated.squeeze() > 0

    def _build_finetune_batch(
        self, view_idx: int
    ) -> Optional[Dict[str, Any]]:
        """
        Build a minimal camera batch for a cached training view.

        This constructs the batch dict required by the forward pass
        to render language features from a specific viewpoint.

        Args:
            view_idx: Index of the cached training view.

        Returns:
            Batch dict with camera parameters, or None if unavailable.
        """
        # Try to use cached batches if available
        if hasattr(self, "cached_batches") and self.cached_batches:
            for batch in self.cached_batches:
                idx = batch.get("index")
                if idx is not None:
                    batch_idx = idx.item() if isinstance(idx, torch.Tensor) else idx
                    if batch_idx == view_idx:
                        return {
                            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()
                        }

        # Fallback: find the cached batch whose index is closest to view_idx
        # (never override index with mismatched camera params)
        if hasattr(self, "cached_batches") and self.cached_batches:
            best_batch = None
            best_dist = float("inf")
            for batch in self.cached_batches:
                idx = batch.get("index")
                if idx is not None:
                    batch_idx = idx.item() if isinstance(idx, torch.Tensor) else idx
                    dist = abs(batch_idx - view_idx)
                    if dist < best_dist:
                        best_dist = dist
                        best_batch = batch
            if best_batch is not None:
                logger.debug(
                    "[_build_finetune_batch] Exact match not found for view %d, "
                    "using nearest cached view (dist=%d)",
                    view_idx, best_dist,
                )
                return {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in best_batch.items()
                }

        return None

    def outpaint(self, batch: Dict[str, Any]) -> None:
        """
        Outpainting to extend scene
        (Placeholder - full implementation would require depth estimation and inpainting)
        """
        # Full implementation would involve:
        # 1. Render current views
        # 2. Identify empty regions
        # 3. Use depth estimation (e.g., GeoWizard)
        # 4. Use inpainting (e.g., Stable Diffusion Inpaint)
        # 5. Add new points to Gaussian model
        pass
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint with autoencoder"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'global_step': self.global_step,
            'geometry_state_dict': {
                '_xyz': self.geometry._xyz.data,
                '_features_dc': self.geometry._features_dc.data,
                '_features_rest': self.geometry._features_rest.data,
                '_scaling': self.geometry._scaling.data,
                '_rotation': self.geometry._rotation.data,
                '_opacity': self.geometry._opacity.data,
                '_language_feature': self.geometry._language_feature.data,
            },
            'autoencoder_state_dict': self.sam_clip_ae.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'config': self.cfg,
        }
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint with autoencoder"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.global_step = checkpoint.get('global_step', 0)
        
        # Load geometry parameters
        for key, value in checkpoint['geometry_state_dict'].items():
            setattr(self.geometry, key, nn.Parameter(value.to(self.device).requires_grad_(True)))
        
        # Load autoencoder (critical for semantic visualization)
        if 'autoencoder_state_dict' in checkpoint:
            self.sam_clip_ae.load_state_dict(checkpoint['autoencoder_state_dict'])
            print("Loaded autoencoder weights")
        
        # Load optimizer
        if self.optimizer and checkpoint.get('optimizer_state_dict'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def save_ply(self, path: str):
        """Save Gaussians to PLY file"""
        self.geometry.save_ply(path)
    
    def load_ply(self, path: str):
        """Load Gaussians from PLY file"""
        self.geometry.load_ply(path)
