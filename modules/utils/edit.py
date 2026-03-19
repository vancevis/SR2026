"""
Scene Editing Module
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, Any
from scipy.spatial.transform import Rotation as R_scipy
import logging
from PIL import Image

try:
    from modules.geometry.inpainting import (
        BackgroundInpainter,
        dilate_mask,
        blur_mask_edges,
        image_to_pointcloud_simple
    )
    INPAINTING_AVAILABLE = True
except ImportError:
    INPAINTING_AVAILABLE = False
    logging.warning("Inpainting module not available")


# Setup logging in 3DitScene style
logger = logging.getLogger(__name__)


def rotation_matrix(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    """
    Create rotation matrix from Euler angles (in degrees)
    
    Args:
        roll: Rotation around X-axis (degrees)
        pitch: Rotation around Y-axis (degrees)
        yaw: Rotation around Z-axis (degrees)
    
    Returns:
        3x3 rotation matrix
    """
    roll = torch.tensor(roll * np.pi / 180.0)
    pitch = torch.tensor(pitch * np.pi / 180.0)
    yaw = torch.tensor(yaw * np.pi / 180.0)
    
    # Roll (X-axis)
    Rx = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(roll), -torch.sin(roll)],
        [0, torch.sin(roll), torch.cos(roll)]
    ], dtype=torch.float32)
    
    # Pitch (Y-axis)
    Ry = torch.tensor([
        [torch.cos(pitch), 0, torch.sin(pitch)],
        [0, 1, 0],
        [-torch.sin(pitch), 0, torch.cos(pitch)]
    ], dtype=torch.float32)
    
    # Yaw (Z-axis)
    Rz = torch.tensor([
        [torch.cos(yaw), -torch.sin(yaw), 0],
        [torch.sin(yaw), torch.cos(yaw), 0],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    # Combined rotation: Rz * Ry * Rx
    R = Rz @ Ry @ Rx
    return R


def build_rotation(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternions to rotation matrices
    
    Args:
        quaternions: [N, 4] quaternion tensor (w, x, y, z)
    
    Returns:
        [N, 3, 3] rotation matrices
    """
    N = quaternions.shape[0]
    
    # Normalize quaternions
    q = quaternions / (quaternions.norm(dim=1, keepdim=True) + 1e-9)
    
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    # Build rotation matrix
    R = torch.zeros(N, 3, 3, device=quaternions.device, dtype=quaternions.dtype)
    
    R[:, 0, 0] = 1 - 2*(y**2 + z**2)
    R[:, 0, 1] = 2*(x*y - w*z)
    R[:, 0, 2] = 2*(x*z + w*y)
    
    R[:, 1, 0] = 2*(x*y + w*z)
    R[:, 1, 1] = 1 - 2*(x**2 + z**2)
    R[:, 1, 2] = 2*(y*z - w*x)
    
    R[:, 2, 0] = 2*(x*z - w*y)
    R[:, 2, 1] = 2*(y*z + w*x)
    R[:, 2, 2] = 1 - 2*(x**2 + y**2)
    
    return R


# Reorder matrix for quaternion conversion (3DitScene convention)
REORDER_MTX = torch.tensor([
    [0, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0]
], dtype=torch.float32)


class SceneEditor:
    """
    Scene Editor for language-guided 3D Gaussian editing
    Supports translate, rotate, scale, delete operations
    With background inpainting support
    """
    
    def __init__(self, system, device='cuda', enable_inpainting=True):
        """
        Initialize SceneEditor
        
        Args:
            system: SceneLangSystem instance
            device: Device to run on
            enable_inpainting: Whether to enable background inpainting for delete operations
        """
        self.system = system
        self.geometry = system.geometry
        self.sam_clip = system.sam_clip
        self.device = device
        
        # Stack-based backup for multi-level undo
        self._backup_stack: list = []
        self._max_undo_levels = 5
        
        # Background inpainting
        self.inpainter = None
        self.inpainting_enabled = False
        if enable_inpainting and INPAINTING_AVAILABLE:
            self.inpainter = BackgroundInpainter()
            if self.inpainter.initialize():
                self.inpainting_enabled = True
                logger.info("Background inpainting enabled")
            else:
                logger.warning("Failed to initialize background inpainting")
        else:
            logger.info("Background inpainting disabled")
        
        logger.info("SceneEditor initialized")
    
    def select_object_by_prompt(
        self,
        prompt: str,
        threshold: float = 0.5,
        uncertainty_aware: bool = True,
        num_augmentations: int = 5,
    ) -> torch.Tensor:
        """
        Select object based on language prompt using semantic features.

        When uncertainty_aware=True (Direction A), computes selection
        confidence via prompt augmentation ensemble:
          1. Generate N slightly varied prompts (e.g., "a red chair",
             "the red chair", "red-colored chair").
          2. Compute relevancy for each variant.
          3. Use mean relevancy for selection, std as uncertainty.
          4. Apply adaptive threshold: select Gaussians where
             mean_relevancy > threshold AND uncertainty < 0.2.

        This reduces false positives from ambiguous CLIP embeddings
        and provides a confidence score for downstream reasoning.

        Args:
            prompt: Text description of target object.
            threshold: Similarity threshold for selection.
            uncertainty_aware: If True, use ensemble-based selection.
            num_augmentations: Number of prompt augmentations for ensemble.

        Returns:
            Boolean mask of selected Gaussians [N].
        """
        logger.info(f"[SceneEditor] Selecting object: '{prompt}'")

        # Get language features from Gaussians
        lang_features = self.geometry.get_language_feature  # [N, 3]

        with torch.no_grad():
            decoded = self.system.sam_clip_ae.decode(lang_features)
            decoded = decoded / (decoded.norm(dim=-1, keepdim=True) + 1e-9)

        if not uncertainty_aware or num_augmentations <= 1:
            # Original single-prompt path
            self.system.sam_clip.model.set_positives([prompt])
            with torch.no_grad():
                relevancy = self.system.sam_clip.model.get_relevancy(decoded, positive_id=0)
                relevancy = relevancy[:, 0]  # [N]
            mask = relevancy > threshold
            self._last_relevancy = relevancy
            self._last_uncertainty = None
        else:
            # Uncertainty-aware ensemble selection (Direction A)
            augmented_prompts = self._augment_prompt(prompt, num_augmentations)
            all_relevancies = []

            for aug_prompt in augmented_prompts:
                self.system.sam_clip.model.set_positives([aug_prompt])
                with torch.no_grad():
                    rel = self.system.sam_clip.model.get_relevancy(decoded, positive_id=0)
                    all_relevancies.append(rel[:, 0])

            # Stack: [num_aug, N]
            rel_stack = torch.stack(all_relevancies, dim=0)
            mean_rel = rel_stack.mean(dim=0)   # [N]
            std_rel = rel_stack.std(dim=0)      # [N] — uncertainty

            # Adaptive selection: high relevancy AND low uncertainty
            uncertainty_threshold = 0.2
            mask = (mean_rel > threshold) & (std_rel < uncertainty_threshold)

            self._last_relevancy = mean_rel
            self._last_uncertainty = std_rel

            logger.info(
                "[SceneEditor] Uncertainty-aware selection: "
                "mean_rel range [%.3f, %.3f], mean_std=%.4f",
                mean_rel.min().item(),
                mean_rel.max().item(),
                std_rel.mean().item(),
            )

        # Direction D: Graph bilateral smoothing on relevancy scores
        # Treats relevancy as a signal on the 3D Gaussian graph and applies
        # a bilateral filter: spatial proximity × feature similarity.
        # Produces spatially coherent selections with clean boundaries.
        relevancy_raw = self._last_relevancy
        smoothed_rel = self._graph_bilateral_smooth(
            relevancy_raw, lang_features, k=16,
            sigma_spatial=None,  # auto from k-NN distance
            sigma_feature=0.3,
        )
        mask = smoothed_rel > threshold

        # Remove small connected components (isolated noise)
        mask = self._remove_small_components(mask, min_size=10)

        self._last_relevancy = smoothed_rel

        num_selected = mask.sum().item()
        logger.info(f"[SceneEditor] Selected {num_selected} / {len(mask)} Gaussians")

        return mask

    def _graph_bilateral_smooth(
        self,
        relevancy: torch.Tensor,
        lang_features: torch.Tensor,
        k: int = 16,
        sigma_spatial: Optional[float] = None,
        sigma_feature: float = 0.3,
    ) -> torch.Tensor:
        """
        Bilateral filter on 3D Gaussian relevancy scores (Direction D).

        Smooths the relevancy signal on a k-NN graph where edge weights
        combine spatial proximity and semantic feature similarity:

            w(i,j) = exp(-d_spatial²/2σ_s²) · exp(-d_feature²/2σ_f²)
            r_smooth[i] = Σ_j w(i,j) · r[j] / Σ_j w(i,j)

        This produces spatially coherent selections while preserving
        sharp boundaries between semantically different regions.

        Args:
            relevancy: Raw relevancy scores [N].
            lang_features: Compressed language features [N, d] (for
                           feature-space distance).
            k: Number of nearest neighbors per Gaussian.
            sigma_spatial: Spatial bandwidth. If None, auto-computed
                           as median k-NN distance.
            sigma_feature: Feature-space bandwidth.

        Returns:
            Smoothed relevancy scores [N].
        """
        N = relevancy.shape[0]
        if N < k + 1:
            return relevancy

        xyz = self.geometry.get_xyz.detach()  # [N, 3]

        # Use CPU FAISS/cKDTree to avoid OOM for large scenes
        try:
            from scipy.spatial import cKDTree
            xyz_np = xyz.cpu().numpy()
            tree = cKDTree(xyz_np)
            # Query k+1 neighbors because the first one is the point itself
            topk_dists_np, topk_idx_np = tree.query(xyz_np, k=k+1, workers=-1)
            
            # Remove self (usually the first neighbor)
            topk_dists = torch.from_numpy(topk_dists_np[:, 1:]).to(self.device)
            topk_idx = torch.from_numpy(topk_idx_np[:, 1:]).to(self.device)
        except ImportError:
            logger.warning("[SceneEditor] scipy.spatial.cKDTree not found, falling back to chunked PyTorch cdist (may be slow)")
            # Chunked exact distance calculation (slower but safe from OOM)
            chunk_size = 4096
            topk_dists = torch.zeros((N, k), device=self.device)
            topk_idx = torch.zeros((N, k), dtype=torch.long, device=self.device)
            
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                chunk_xyz = xyz[start:end]
                dists = torch.cdist(chunk_xyz, xyz)
                dists[torch.arange(end - start, device=self.device), torch.arange(start, end, device=self.device)] = float("inf")
                dists_k, idx_k = dists.topk(k, dim=1, largest=False)
                topk_dists[start:end] = dists_k
                topk_idx[start:end] = idx_k

        smoothed = torch.zeros_like(relevancy)

        # Precompute normalized features for cosine distance
        feat_norm = lang_features.detach()
        feat_norm = feat_norm / (feat_norm.norm(dim=-1, keepdim=True) + 1e-9)

        # Auto sigma_spatial from median k-NN distance
        if sigma_spatial is None:
            sigma_s = topk_dists.median().item() + 1e-9
        else:
            sigma_s = sigma_spatial

        # Spatial weights: exp(-d² / 2σ²)
        w_spatial = torch.exp(-topk_dists ** 2 / (2 * sigma_s ** 2))  # [N, k]
        
        # Process feature distances and smooth in chunks to save memory
        chunk_size = 8192
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            
            chunk_feat = feat_norm[start:end]  # [C, d]
            chunk_topk_idx = topk_idx[start:end]  # [C, k]
            
            neighbor_feat = feat_norm[chunk_topk_idx.reshape(-1)].reshape(
                end - start, k, -1
            )  # [C, k, d]
            
            feat_diff = (
                chunk_feat.unsqueeze(1) - neighbor_feat
            ).norm(dim=-1)  # [C, k]
            
            w_feature = torch.exp(
                -feat_diff ** 2 / (2 * sigma_feature ** 2)
            )  # [C, k]

            # Combined bilateral weight
            w = w_spatial[start:end] * w_feature  # [C, k]

            # Weighted average of neighbor relevancy
            neighbor_rel = relevancy[chunk_topk_idx.reshape(-1)].reshape(
                end - start, k
            )  # [C, k]
            smoothed[start:end] = (
                (w * neighbor_rel).sum(dim=1) / (w.sum(dim=1) + 1e-9)
            )

        logger.info(
            "[SceneEditor] Graph bilateral smooth: σ_s=auto, σ_f=%.2f, k=%d",
            sigma_feature, k,
        )
        return smoothed

    def _remove_small_components(
        self,
        mask: torch.Tensor,
        min_size: int = 10,
        connectivity_radius: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Remove small spatially isolated clusters from a selection mask.

        Uses a simple greedy flood-fill on the 3D positions: two selected
        Gaussians are "connected" if their distance is below a threshold.
        Components with fewer than min_size members are removed.

        Args:
            mask: Boolean selection mask [N].
            min_size: Minimum component size to keep.
            connectivity_radius: Distance threshold for connectivity.
                If None, auto-computed as 2× median nearest-neighbor
                distance among selected Gaussians.

        Returns:
            Cleaned boolean mask [N].
        """
        selected_idx = torch.where(mask)[0]
        M = selected_idx.shape[0]
        if M <= min_size:
            return mask

        xyz_sel = self.geometry.get_xyz.detach()[selected_idx]  # [M, 3]

        # Auto connectivity radius
        if connectivity_radius is None:
            if M > 1:
                dists = torch.cdist(xyz_sel, xyz_sel)
                dists.fill_diagonal_(float("inf"))
                nn_dists = dists.min(dim=1).values
                connectivity_radius = 2.0 * nn_dists.median().item()
            else:
                return mask

        # Build adjacency and find connected components via BFS
        dists = torch.cdist(xyz_sel, xyz_sel)  # [M, M]
        adj = dists < connectivity_radius
        visited = torch.zeros(M, dtype=torch.bool, device=self.device)
        component_labels = torch.full((M,), -1, dtype=torch.long, device=self.device)
        comp_id = 0

        for seed in range(M):
            if visited[seed]:
                continue
            # BFS
            queue = [seed]
            visited[seed] = True
            component_labels[seed] = comp_id
            head = 0
            while head < len(queue):
                node = queue[head]
                head += 1
                neighbors = torch.where(adj[node] & ~visited)[0]
                for nb in neighbors.tolist():
                    visited[nb] = True
                    component_labels[nb] = comp_id
                    queue.append(nb)
            comp_id += 1

        # Keep only components >= min_size
        cleaned_mask = mask.clone()
        for cid in range(comp_id):
            comp_members = (component_labels == cid)
            if comp_members.sum().item() < min_size:
                cleaned_mask[selected_idx[comp_members]] = False

        removed = mask.sum().item() - cleaned_mask.sum().item()
        if removed > 0:
            logger.info(
                "[SceneEditor] Removed %d isolated Gaussians "
                "(%d components below min_size=%d)",
                removed, comp_id, min_size,
            )
        return cleaned_mask

    @staticmethod
    def _augment_prompt(prompt: str, n: int) -> list:
        """
        Generate augmented prompt variants for ensemble selection.

        Uses deterministic template-based augmentation to avoid
        requiring an LLM call at selection time.

        Args:
            prompt: Original text prompt.
            n: Number of variants to generate (including original).

        Returns:
            List of augmented prompt strings.
        """
        templates = [
            "{prompt}",
            "a {prompt}",
            "the {prompt}",
            "a photo of {prompt}",
            "a photo of a {prompt}",
            "{prompt} in a scene",
            "a 3D rendering of {prompt}",
            "{prompt}, realistic",
        ]
        variants = []
        for i in range(min(n, len(templates))):
            variants.append(templates[i].format(prompt=prompt))
        return variants
    
    def backup_parameters(self):
        """Push current Gaussian parameters onto the undo stack (all attributes)."""
        snapshot = {
            "_xyz": self.geometry._xyz.data.clone(),
            "_rotation": self.geometry._rotation.data.clone(),
            "_opacity": self.geometry._opacity.data.clone(),
            "_scaling": self.geometry._scaling.data.clone(),
            "_features_dc": self.geometry._features_dc.data.clone(),
            "_features_rest": self.geometry._features_rest.data.clone(),
            "_language_feature": self.geometry._language_feature.data.clone(),
        }
        self._backup_stack.append(snapshot)
        # Evict oldest snapshot if stack exceeds limit
        if len(self._backup_stack) > self._max_undo_levels:
            self._backup_stack.pop(0)
        logger.info(
            "[SceneEditor] Parameters backed up (stack depth=%d)",
            len(self._backup_stack),
        )

    def restore_parameters(self):
        """Pop the most recent snapshot from the undo stack and restore it."""
        if not self._backup_stack:
            logger.warning("[SceneEditor] No backup to restore")
            return
        snapshot = self._backup_stack.pop()
        for attr, tensor in snapshot.items():
            setattr(self.geometry, attr, nn.Parameter(tensor.to(self.device)))
        logger.info(
            "[SceneEditor] Parameters restored (stack depth=%d)",
            len(self._backup_stack),
        )
    
    def translate_object(
        self, 
        mask: torch.Tensor, 
        offset: Tuple[float, float, float],
        inplace: bool = True
    ):
        """
        Translate selected Gaussians
        
        Args:
            mask: Boolean selection mask [N]
            offset: (x, y, z) translation offset
            inplace: Whether to modify parameters in-place
        """
        if mask.sum() == 0:
            logger.warning("[SceneEditor] Empty mask! No object found to translate.")
            return

        logger.info(f"[SceneEditor] Translating object by offset {offset}")
        
        if not inplace:
            self.backup_parameters()
        
        # Apply translation only to masked Gaussians
        new_xyz = self.geometry._xyz.data.clone()
        new_xyz[mask] += torch.tensor(offset, device=self.device, dtype=new_xyz.dtype)
        self.geometry._xyz = nn.Parameter(new_xyz)
        
        logger.info(f"[SceneEditor] Translation complete")
    
    def rotate_object(
        self,
        mask: torch.Tensor,
        rotation: Tuple[float, float, float],
        center: Optional[torch.Tensor] = None,
        inplace: bool = True
    ):
        """
        Rotate selected Gaussians around a center point
        
        Args:
            mask: Boolean selection mask [N]
            rotation: (roll, pitch, yaw) in degrees
            center: Center of rotation [3]. If None, use object center
            inplace: Whether to modify parameters in-place
        """
        if mask.sum() == 0:
            logger.warning("[SceneEditor] Empty mask! No object found to rotate.")
            return

        logger.info(f"[SceneEditor] Rotating object by {rotation} degrees")
        
        if not inplace:
            self.backup_parameters()
        
        # Get rotation matrix
        rot_matrix = rotation_matrix(rotation[0], rotation[1], rotation[2]).to(self.device)
        
        # Get selected points (clone to avoid in-place mutation of Parameter data)
        prev_xyz = self.geometry.get_xyz.data.clone()
        ooi_xyz = prev_xyz[mask]
        
        # Calculate center
        if center is None:
            center = ooi_xyz.mean(0)
        
        # Rotate positions
        ooi_xyz_centered = ooi_xyz - center
        after_xyz = torch.einsum('ab,nb->na', rot_matrix, ooi_xyz_centered) + center
        prev_xyz[mask] = after_xyz
        self.geometry._xyz = nn.Parameter(prev_xyz)
        
        # Rotate quaternions (only for masked Gaussians to avoid O(N) overhead)
        prev_rotation = self.geometry.get_rotation.data.clone()
        masked_quats = prev_rotation[mask]  # [M, 4]
        masked_rot_mtx = build_rotation(masked_quats)  # [M, 3, 3]
        after_rot_mtx = torch.einsum('ab,nbc->nac', rot_matrix, masked_rot_mtx)
        
        # Convert back to quaternions (only M, not N)
        after_rot_scipy = R_scipy.from_matrix(after_rot_mtx.detach().cpu().numpy())
        after_quats = torch.from_numpy(after_rot_scipy.as_quat()).to(self.device).float()
        
        # Reorder (xyzw -> wxyz)
        after_quats = torch.einsum('ab,nb->na', REORDER_MTX.to(self.device), after_quats)
        prev_rotation[mask] = after_quats
        self.geometry._rotation = nn.Parameter(prev_rotation)
        
        logger.info(f"[SceneEditor] Rotation complete")
    
    def scale_object(
        self,
        mask: torch.Tensor,
        scale_factor: float,
        center: Optional[torch.Tensor] = None,
        inplace: bool = True
    ):
        """
        Scale selected Gaussians (both sizes and positions relative to center)
        
        Args:
            mask: Boolean selection mask [N]
            scale_factor: Scaling factor (e.g., 1.5 = 150%)
            center: Center of scaling [3]. If None, use object centroid.
            inplace: Whether to modify parameters in-place
        """
        if mask.sum() == 0:
            logger.warning("[SceneEditor] Empty mask! No object found to scale.")
            return

        if scale_factor <= 0:
            raise ValueError(f"scale_factor must be positive, got {scale_factor}")
        
        logger.info(f"[SceneEditor] Scaling object by factor {scale_factor}")
        
        if not inplace:
            self.backup_parameters()
        
        # Scale positions relative to object center (clone to avoid in-place mutation)
        prev_xyz = self.geometry.get_xyz.data.clone()
        ooi_xyz = prev_xyz[mask]
        if center is None:
            center = ooi_xyz.mean(0)
        ooi_xyz_centered = ooi_xyz - center
        after_xyz = ooi_xyz_centered * scale_factor + center
        prev_xyz[mask] = after_xyz
        self.geometry._xyz = nn.Parameter(prev_xyz)
        
        # Scale the Gaussian covariance sizes (operate in log-space directly)
        # NOTE: _scaling is stored in log-space; get_scaling returns exp(_scaling).
        # We must add log(scale_factor) in log-space, NOT in activated space.
        raw_scaling = self.geometry._scaling.data.clone()
        raw_scaling[mask] = raw_scaling[mask] + np.log(scale_factor)
        self.geometry._scaling = nn.Parameter(raw_scaling)
        
        logger.info(f"[SceneEditor] Scaling complete")
    
    def delete_object(
        self,
        mask: torch.Tensor,
        inpaint_background: bool = True,
        background_prompt: str = "natural background, photorealistic",
        prune: bool = False
    ):
        """
        Delete selected Gaussians and optionally inpaint background
        
        Args:
            mask: Boolean selection mask [N]
            inpaint_background: Whether to inpaint the background (requires diffusers)
            background_prompt: Text prompt for background inpainting
            prune: If True, physically remove Gaussians; if False, only zero opacity
        """
        if mask.sum() == 0:
            logger.warning("[SceneEditor] Empty mask! No object found to delete.")
            return

        num_deleted = mask.sum().item()
        logger.info(f"Deleting {num_deleted} Gaussians (prune={prune})")
        
        # Save deleted Gaussian positions BEFORE pruning for inpainting projection
        deleted_positions = self.geometry.get_xyz.data[mask].clone()
        
        if prune:
            keep = ~mask
            self.geometry._xyz = nn.Parameter(self.geometry._xyz.data[keep])
            self.geometry._features_dc = nn.Parameter(self.geometry._features_dc.data[keep])
            self.geometry._features_rest = nn.Parameter(self.geometry._features_rest.data[keep])
            self.geometry._scaling = nn.Parameter(self.geometry._scaling.data[keep])
            self.geometry._rotation = nn.Parameter(self.geometry._rotation.data[keep])
            self.geometry._opacity = nn.Parameter(self.geometry._opacity.data[keep])
            self.geometry._language_feature = nn.Parameter(self.geometry._language_feature.data[keep])
        else:
            prev_opacity = self.geometry._opacity.data
            prev_opacity[mask] = -10.0  # log(sigmoid(-10)) ≈ 0
            self.geometry._opacity = nn.Parameter(prev_opacity)
        
        # Background inpainting (use saved positions, not stale mask)
        if inpaint_background and self.inpainting_enabled:
            logger.info("Attempting background inpainting...")
            try:
                self._inpaint_deleted_region(deleted_positions, background_prompt)
            except Exception as e:
                logger.error(f"Background inpainting failed: {e}")
                logger.info("Continuing without background inpainting")
        elif inpaint_background:
            logger.warning("Background inpainting requested but not available")
            logger.warning("Install with: pip install diffusers transformers accelerate")
        
        logger.info("Deletion complete")
    
    def _inpaint_deleted_region(self, deleted_positions: torch.Tensor, prompt: str):
        """
        Inpaint the region where Gaussians were deleted.
        
        Args:
            deleted_positions: 3D positions of deleted Gaussians [M, 3].
                               Must be saved BEFORE pruning.
            prompt: Background prompt for inpainting.
        """
        logger.info("Rendering scene for inpainting...")
        
        # Get a reference view from the dataset
        # Use the first training view as reference
        if not hasattr(self.system, 'trainer') or self.system.trainer is None:
            logger.warning("No trainer available, skipping inpainting")
            return
        
        try:
            dataloader = self.system.trainer.train_dataloader
            if dataloader is None:
                logger.warning("No dataloader available")
                return
            
            # Get first batch
            sample = next(iter(dataloader()))
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    sample[k] = v.to(self.device)
            
            # Render current scene (with deleted object)
            with torch.no_grad():
                output = self.system(sample)
                rendered_rgb = output['comp_rgb'][0]  # [H, W, 3]
            
            # Convert to PIL Image
            rgb_np = (rendered_rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(rgb_np)
            
            # Create mask image by projecting deleted Gaussians to 2D
            h, w = rgb_np.shape[:2]
            mask_2d = np.zeros((h, w), dtype=np.uint8)
            
            # Extract camera matrices from sample batch
            c2w = sample.get('c2w')
            if c2w is not None:
                c2w_mat = c2w[0] if c2w.dim() == 3 else c2w  # [4,4]
                w2c = torch.inverse(c2w_mat)  # [4,4]
                
                # Get FoV for projection
                fovx = sample.get('fovx', sample.get('fov', 0.8))
                fovy = sample.get('fovy', fovx)
                if isinstance(fovx, torch.Tensor):
                    fovx = fovx.item()
                if isinstance(fovy, torch.Tensor):
                    fovy = fovy.item()
                
                fx = w / (2.0 * np.tan(fovx / 2.0))
                fy = h / (2.0 * np.tan(fovy / 2.0))
                cx, cy = w / 2.0, h / 2.0
                
                # Project: world → camera → image
                pts_h = torch.cat([
                    deleted_positions,
                    torch.ones(deleted_positions.shape[0], 1, device=self.device)
                ], dim=1)  # [M, 4]
                pts_cam = (w2c @ pts_h.T).T[:, :3]  # [M, 3]
                
                # Filter points in front of camera
                valid = pts_cam[:, 2] > 0.01
                if valid.sum() > 0:
                    pts_cam = pts_cam[valid]
                    u = (fx * pts_cam[:, 0] / pts_cam[:, 2] + cx).long().cpu().numpy()
                    v = (fy * pts_cam[:, 1] / pts_cam[:, 2] + cy).long().cpu().numpy()
                    
                    # Splat each projected point with a small radius
                    splat_radius = max(3, min(h, w) // 100)
                    for ui, vi in zip(u, v):
                        if 0 <= ui < w and 0 <= vi < h:
                            r_min = max(0, vi - splat_radius)
                            r_max = min(h, vi + splat_radius + 1)
                            c_min = max(0, ui - splat_radius)
                            c_max = min(w, ui + splat_radius + 1)
                            mask_2d[r_min:r_max, c_min:c_max] = 255
            else:
                # Fallback: render with zero-opacity and diff against original
                logger.warning("No c2w in batch, using convex hull fallback for mask")
                center_y, center_x = h // 2, w // 2
                radius = min(h, w) // 4
                y, x = np.ogrid[:h, :w]
                circle_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
                mask_2d[circle_mask] = 255
            
            # Dilate mask slightly
            mask_dilated = dilate_mask(mask_2d, kernel_size=11)
            mask_img = Image.fromarray((mask_dilated * 255).astype(np.uint8), mode='L')
            
            # Run inpainting
            logger.info(f"Running inpainting with prompt: '{prompt}'")
            inpainted = self.inpainter.inpaint(
                image=image,
                mask=mask_img,
                prompt=prompt,
                guidance_scale=7.5,
                num_inference_steps=30  # Reduced for speed
            )
            
            # Convert inpainted image to point cloud
            logger.info("Converting inpainted background to 3D points...")
            camera_params = {
                'fov': sample.get('fov', 60.0) if isinstance(sample.get('fov'), (int, float)) else 60.0
            }
            
            points, colors = image_to_pointcloud_simple(
                image=inpainted,
                mask=mask_img,
                depth_estimate=5.0,  # Rough depth estimate
                fov=camera_params['fov']
            )
            
            if points.shape[0] > 0:
                # Sample to avoid too many points
                max_bg_points = 5000
                if points.shape[0] > max_bg_points:
                    indices = np.random.choice(points.shape[0], max_bg_points, replace=False)
                    points = points[indices]
                    colors = colors[indices]
                
                logger.info(f"Adding {points.shape[0]} background points")
                
                # Merge background points into scene
                from modules.geometry.gaussian_base import BasicPointCloud, RGB2SH
                
                pcd = BasicPointCloud(
                    points=points,
                    colors=colors,
                    normals=np.zeros_like(points)
                )
                
                # Add points using existing method
                new_xyz = torch.tensor(pcd.points).float().to(self.device)
                new_color_sh = RGB2SH(torch.tensor(pcd.colors).float().to(self.device))
                
                # Initialize features for new points
                new_features = torch.zeros((new_xyz.shape[0], 3, (self.geometry.max_sh_degree + 1) ** 2)).float().to(self.device)
                new_features[:, :3, 0] = new_color_sh
                
                # Use median scale from existing Gaussians
                existing_scales = self.geometry.get_scaling
                median_scale = existing_scales.median(dim=0)[0]
                new_scales = torch.log(median_scale.repeat(new_xyz.shape[0], 1))
                
                # Identity rotations
                new_rots = torch.zeros((new_xyz.shape[0], 4), device=self.device)
                new_rots[:, 0] = 1
                
                # Low opacity initially
                from modules.geometry.gaussian_base import inverse_sigmoid
                new_opacities = inverse_sigmoid(
                    0.05 * torch.ones((new_xyz.shape[0], 1), dtype=torch.float, device=self.device)
                )
                
                # Zero language features
                new_lang_features = torch.zeros((new_xyz.shape[0], self.geometry.cfg.lang_feature_dim), device=self.device)
                
                # Concatenate with existing Gaussians
                self.geometry._xyz = nn.Parameter(torch.cat([self.geometry._xyz.data, new_xyz], dim=0))
                self.geometry._features_dc = nn.Parameter(torch.cat([
                    self.geometry._features_dc.data,
                    new_features[:, :, 0:1].transpose(1, 2).contiguous()
                ], dim=0))
                self.geometry._features_rest = nn.Parameter(torch.cat([
                    self.geometry._features_rest.data,
                    new_features[:, :, 1:].transpose(1, 2).contiguous()
                ], dim=0))
                self.geometry._scaling = nn.Parameter(torch.cat([self.geometry._scaling.data, new_scales], dim=0))
                self.geometry._rotation = nn.Parameter(torch.cat([self.geometry._rotation.data, new_rots], dim=0))
                self.geometry._opacity = nn.Parameter(torch.cat([self.geometry._opacity.data, new_opacities], dim=0))
                self.geometry._language_feature = nn.Parameter(torch.cat([self.geometry._language_feature.data, new_lang_features], dim=0))
                
                logger.info(f"Background inpainting complete. Total Gaussians: {self.geometry._xyz.shape[0]}")
            else:
                logger.warning("No background points generated")
                
        except Exception as e:
            logger.error(f"Error during inpainting: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def edit_scene(
        self,
        prompt: str,
        operation: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        High-level editing interface
        
        Args:
            prompt: Language description of target object
            operation: One of ['translate', 'rotate', 'scale', 'delete']
            **kwargs: Operation-specific parameters
                - For translate: offset=(x, y, z)
                - For rotate: rotation=(roll, pitch, yaw), center=None
                - For scale: scale_factor=1.5
        
        Returns:
            Dict with editing results
        """
        logger.info(f"Starting edit operation: {operation}")
        logger.info(f"Target object: '{prompt}'")
        
        # Select object
        threshold = kwargs.get('threshold', 0.5)
        mask = self.select_object_by_prompt(prompt, threshold)
        
        if mask.sum() == 0:
            logger.warning("[SceneEditor] No Gaussians selected! Try lowering threshold.")
            return {'success': False, 'message': 'No object found'}
        
        # Execute operation
        if operation == 'translate':
            offset = kwargs.get('offset', (0.0, 0.0, 0.0))
            self.translate_object(mask, offset)
        
        elif operation == 'rotate':
            rotation = kwargs.get('rotation', (0.0, 0.0, 0.0))
            center = kwargs.get('center', None)
            self.rotate_object(mask, rotation, center)
        
        elif operation == 'scale':
            scale_factor = kwargs.get('scale_factor', 1.0)
            self.scale_object(mask, scale_factor)
        
        elif operation == 'delete':
            self.delete_object(mask)
        
        else:
            logger.error(f"[SceneEditor] Unknown operation: {operation}")
            return {'success': False, 'message': f'Unknown operation: {operation}'}
        
        # Post-edit semantic field update
        # Default is now False (Direction A: agent decides via evaluate + finetune).
        # Callers that bypass the agent can still opt in explicitly.
        if kwargs.get('semantic_update', False) and operation != 'delete':
            if hasattr(self.system, 'local_semantic_finetune'):
                finetune_steps = kwargs.get('finetune_steps', 30)
                logger.info(
                    "[SceneEditor] Running post-edit semantic finetune "
                    "(%d steps)...", finetune_steps
                )
                self.system.local_semantic_finetune(
                    affected_mask=mask,
                    num_steps=finetune_steps,
                )
        
        logger.info(f"Edit operation '{operation}' completed successfully")
        
        # Compute object center for richer feedback
        object_center = None
        if mask.sum() > 0:
            object_center = (
                self.geometry.get_xyz[mask]
                .mean(dim=0)
                .cpu()
                .tolist()
            )
        
        return {
            'success': True,
            'operation': operation,
            'num_selected': mask.sum().item(),
            'object_center': object_center,
            'total_gaussians': len(mask),
            'mask': mask,
        }
