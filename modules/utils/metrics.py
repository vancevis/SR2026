"""
Semantic Editing Metrics Module

Provides quantitative metrics for evaluating semantic consistency
before and after 3D scene editing operations. These metrics form
the core of the closed-loop Semantic State Evaluator, enabling
the LLM agent to reason about edit quality and adaptively adjust
parameters.

Key Metrics:
    - Semantic Consistency Score (SCS): Measures how well the semantic
      field in the edited region aligns with the training-view targets.
    - Unedited Region Preservation (URP): Ensures non-edited regions
      remain semantically stable after an edit.
    - Edit Precision / Recall: Evaluates whether the intended object
      was correctly selected and edited.
    - Feature Distribution Shift (FDS): Detects abnormal feature
      drift caused by editing artifacts.
"""

import math

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SemanticEvalResult:
    """Container for semantic evaluation metrics."""
    semantic_consistency_score: float
    unedited_preservation_score: float
    edit_precision: float
    edit_recall: float
    feature_distribution_shift: float
    recommended_finetune_steps: int
    confidence: float
    # Direction C: Multi-View Consistency
    multi_view_consistency: float = 1.0
    per_view_scs: Optional[Dict[int, float]] = None
    worst_view_idx: int = -1
    worst_view_scs: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "semantic_consistency_score": round(self.semantic_consistency_score, 4),
            "unedited_preservation_score": round(self.unedited_preservation_score, 4),
            "edit_precision": round(self.edit_precision, 4),
            "edit_recall": round(self.edit_recall, 4),
            "feature_distribution_shift": round(self.feature_distribution_shift, 4),
            "recommended_finetune_steps": self.recommended_finetune_steps,
            "confidence": round(self.confidence, 4),
            "multi_view_consistency": round(self.multi_view_consistency, 4),
            "worst_view_scs": round(self.worst_view_scs, 4),
        }

    def summary(self) -> str:
        """One-line human-readable summary for LLM observation."""
        return (
            f"SCS={self.semantic_consistency_score:.3f}, "
            f"URP={self.unedited_preservation_score:.3f}, "
            f"Prec={self.edit_precision:.3f}, "
            f"Rec={self.edit_recall:.3f}, "
            f"FDS={self.feature_distribution_shift:.3f}, "
            f"MVC={self.multi_view_consistency:.3f}, "
            f"worst_view_SCS={self.worst_view_scs:.3f}, "
            f"rec_steps={self.recommended_finetune_steps}, "
            f"conf={self.confidence:.3f}"
        )


def semantic_consistency_score(
    rendered_lang: torch.Tensor,
    target_lang: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """
    Semantic Consistency Score (SCS).

    Measures cosine similarity between rendered and target language
    features, optionally restricted to a 2D spatial mask.

    Args:
        rendered_lang: Rendered language features [H, W, D].
        target_lang: Target language features [H, W, D].
        mask: Optional boolean mask [H, W]. If provided, only compute
              over True pixels.

    Returns:
        SCS in [0, 1], where 1 = perfect consistency.
    """
    if rendered_lang.shape != target_lang.shape:
        target_lang = F.interpolate(
            target_lang.permute(2, 0, 1).unsqueeze(0),
            size=rendered_lang.shape[:2],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).permute(1, 2, 0)

    pred = rendered_lang.reshape(-1, rendered_lang.shape[-1])
    tgt = target_lang.reshape(-1, target_lang.shape[-1])

    if mask is not None:
        flat_mask = mask.reshape(-1)
        pred = pred[flat_mask]
        tgt = tgt[flat_mask]

    if pred.shape[0] == 0:
        return 1.0

    cos_sim = F.cosine_similarity(pred, tgt, dim=-1)
    return cos_sim.mean().item()


def unedited_preservation_score(
    pre_edit_lang: torch.Tensor,
    post_edit_lang: torch.Tensor,
    edited_mask: torch.Tensor,
) -> float:
    """
    Unedited Region Preservation (URP).

    Measures how stable the semantic features remain in the
    *non-edited* region of the scene after an edit.

    Args:
        pre_edit_lang: Language features before edit [H, W, D].
        post_edit_lang: Language features after edit [H, W, D].
        edited_mask: Boolean mask of the edited region [H, W].

    Returns:
        URP in [0, 1], where 1 = no change in unedited region.
    """
    unedited_mask = ~edited_mask
    if unedited_mask.sum() == 0:
        return 1.0

    pre = pre_edit_lang[unedited_mask]
    post = post_edit_lang[unedited_mask]

    cos_sim = F.cosine_similarity(pre, post, dim=-1)
    return cos_sim.mean().item()


def edit_precision_recall(
    selection_mask: torch.Tensor,
    ground_truth_mask: Optional[torch.Tensor] = None,
    relevancy_scores: Optional[torch.Tensor] = None,
    threshold: float = 0.5,
) -> Tuple[float, float]:
    """
    Edit Precision and Recall.

    When ground truth is available, computes standard precision/recall.
    Otherwise, uses relevancy score distribution as a proxy:
      - Precision proxy: mean relevancy of selected Gaussians
      - Recall proxy: fraction of high-relevancy Gaussians captured

    Args:
        selection_mask: Boolean mask of selected Gaussians [N].
        ground_truth_mask: Optional ground-truth object mask [N].
        relevancy_scores: Optional CLIP relevancy scores [N].
        threshold: Relevancy threshold for high-confidence Gaussians.

    Returns:
        (precision, recall) both in [0, 1].
    """
    if ground_truth_mask is not None:
        tp = (selection_mask & ground_truth_mask).sum().float()
        fp = (selection_mask & ~ground_truth_mask).sum().float()
        fn = (~selection_mask & ground_truth_mask).sum().float()

        precision = (tp / (tp + fp + 1e-9)).item()
        recall = (tp / (tp + fn + 1e-9)).item()
        return precision, recall

    if relevancy_scores is not None:
        selected_scores = relevancy_scores[selection_mask]
        precision_proxy = selected_scores.mean().item() if selected_scores.numel() > 0 else 0.0

        high_rel = relevancy_scores > threshold
        if high_rel.sum() > 0:
            recall_proxy = (selection_mask & high_rel).sum().float() / high_rel.sum().float()
            recall_proxy = recall_proxy.item()
        else:
            recall_proxy = 0.0

        return precision_proxy, recall_proxy

    return 0.5, 0.5


def feature_distribution_shift(
    pre_features: torch.Tensor,
    post_features: torch.Tensor,
    region_mask: Optional[torch.Tensor] = None,
) -> float:
    """
    Feature Distribution Shift (FDS).

    Computes the L2 distance between the mean feature vectors
    before and after editing, normalized by the pre-edit feature
    magnitude. A large FDS indicates potential editing artifacts.

    Args:
        pre_features: Pre-edit Gaussian language features [N, D].
        post_features: Post-edit Gaussian language features [N, D].
        region_mask: Optional mask to restrict computation [N].

    Returns:
        FDS >= 0. Values > 0.5 typically indicate significant drift.
    """
    if region_mask is not None:
        pre = pre_features[region_mask]
        post = post_features[region_mask]
    else:
        pre = pre_features
        post = post_features

    if pre.shape[0] == 0:
        return 0.0

    pre_mean = pre.mean(dim=0)
    post_mean = post.mean(dim=0)

    shift = (post_mean - pre_mean).norm().item()
    normalizer = pre_mean.norm().item() + 1e-9
    return shift / normalizer


def compute_recommended_finetune_steps(
    scs: float,
    urp: float,
    fds: float,
    base_steps: int = 30,
    max_steps: int = 200,
) -> int:
    """
    Adaptively compute recommended fine-tuning steps based on metrics.

    The formula scales steps inversely with semantic quality:
        steps = base * (1 + alpha * (1 - SCS) + beta * FDS - gamma * URP)

    High SCS / low FDS → fewer steps; low SCS / high FDS → more steps.

    Args:
        scs: Semantic Consistency Score [0, 1].
        urp: Unedited Region Preservation [0, 1].
        fds: Feature Distribution Shift [0, inf).
        base_steps: Minimum number of steps.
        max_steps: Maximum cap.

    Returns:
        Recommended number of fine-tuning gradient steps.
    """
    alpha = 3.0
    beta = 2.0
    gamma = 1.0

    quality_deficit = alpha * (1.0 - scs) + beta * min(fds, 2.0) - gamma * urp
    quality_deficit = max(quality_deficit, 0.0)

    steps = int(base_steps * (1.0 + quality_deficit))
    return min(steps, max_steps)


class SemanticStateEvaluator:
    """
    Evaluates the semantic state of a 3D scene before and after editing.

    This is the core innovation component: it provides quantitative
    feedback to the LLM agent, enabling closed-loop semantic-aware
    editing decisions.

    Workflow:
        1. Before edit: snapshot pre-edit language features.
        2. After edit: render post-edit features and compute metrics.
        3. Return SemanticEvalResult to the agent for reasoning.

    Attributes:
        system: SceneLangSystem instance.
        device: Compute device.
        pre_edit_features: Cached pre-edit Gaussian language features.
        pre_edit_rendered: Dict of pre-edit rendered language maps.
    """

    def __init__(self, system, device: str = "cuda"):
        self.system = system
        self.device = device
        self.pre_edit_features: Optional[torch.Tensor] = None
        self.pre_edit_rendered: Dict[int, torch.Tensor] = {}

    def snapshot_pre_edit(self, num_views: int = 3) -> None:
        """
        Cache pre-edit state for later comparison.

        Args:
            num_views: Number of views to render for spatial evaluation.
        """
        self.pre_edit_features = (
            self.system.geometry._language_feature.data.clone().detach()
        )

        self.pre_edit_rendered.clear()

        if hasattr(self.system, "cached_batches") and self.system.cached_batches:
            views = self.system.cached_batches[:num_views]
            for i, batch in enumerate(views):
                batch_gpu = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                with torch.no_grad():
                    out = self.system(batch_gpu)
                lang = out.get("lang")
                if lang is not None:
                    self.pre_edit_rendered[i] = lang[0].detach().clone()

        logger.info(
            "[SemanticEvaluator] Snapshot taken: %d Gaussians, %d rendered views",
            self.pre_edit_features.shape[0],
            len(self.pre_edit_rendered),
        )

    def evaluate_post_edit(
        self,
        edit_mask: Optional[torch.Tensor] = None,
        selection_mask: Optional[torch.Tensor] = None,
        relevancy_scores: Optional[torch.Tensor] = None,
    ) -> SemanticEvalResult:
        """
        Evaluate semantic state after an editing operation.

        Args:
            edit_mask: Boolean mask of edited Gaussians [N].
            selection_mask: Boolean mask used for object selection [N].
            relevancy_scores: CLIP relevancy scores used during selection [N].

        Returns:
            SemanticEvalResult with all computed metrics.
        """
        post_features = (
            self.system.geometry._language_feature.data.clone().detach()
        )

        # -- SCS: average across cached views --
        scs_values = []
        if (
            hasattr(self.system, "semantic_targets")
            and self.system.semantic_targets
            and hasattr(self.system, "cached_batches")
            and self.system.cached_batches
        ):
            view_indices = list(self.system.semantic_targets.keys())[:3]
            for view_idx in view_indices:
                batch = self.system._build_finetune_batch(view_idx)
                if batch is None:
                    continue
                with torch.no_grad():
                    out = self.system(batch)
                rendered = out.get("lang")
                if rendered is None:
                    continue
                target = self.system.semantic_targets[view_idx].to(self.device)
                scs_val = semantic_consistency_score(rendered[0], target)
                scs_values.append(scs_val)

        scs = float(np.mean(scs_values)) if scs_values else 0.5

        # Direction C: per-view SCS map and Multi-View Consistency
        per_view_scs: Dict[int, float] = {}
        if scs_values:
            for i, val in enumerate(scs_values):
                if i < len(view_indices):
                    per_view_scs[view_indices[i]] = val
        mvc = 1.0
        worst_view_idx = -1
        worst_view_scs = scs
        if len(scs_values) >= 2:
            scs_std = float(np.std(scs_values))
            scs_mean = float(np.mean(scs_values))
            # MVC = 1 - coefficient_of_variation (clamped to [0, 1])
            mvc = max(0.0, 1.0 - scs_std / (scs_mean + 1e-9))
            worst_idx_local = int(np.argmin(scs_values))
            worst_view_scs = scs_values[worst_idx_local]
            if worst_idx_local < len(view_indices):
                worst_view_idx = view_indices[worst_idx_local]

        # -- URP: stability of unedited region --
        urp = 1.0
        if self.pre_edit_rendered and hasattr(self.system, "cached_batches"):
            urp_values = []
            for i, pre_lang in self.pre_edit_rendered.items():
                if i >= len(self.system.cached_batches):
                    continue
                batch = self.system.cached_batches[i]
                batch_gpu = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                with torch.no_grad():
                    out = self.system(batch_gpu)
                post_lang = out.get("lang")
                if post_lang is None:
                    continue

                H, W = pre_lang.shape[:2]
                edited_2d = self._project_mask_to_2d(
                    edit_mask, batch_gpu, H, W
                )
                urp_val = unedited_preservation_score(
                    pre_lang.to(self.device), post_lang[0], edited_2d
                )
                urp_values.append(urp_val)
            if urp_values:
                urp = float(np.mean(urp_values))

        # -- Precision / Recall --
        prec, rec = 0.5, 0.5
        if selection_mask is not None:
            prec, rec = edit_precision_recall(
                selection_mask,
                relevancy_scores=relevancy_scores,
            )

        # -- FDS --
        # After prune, Gaussian count changes and index mapping breaks.
        # We use two strategies:
        #   (a) If counts match → use per-index FDS with edit_mask.
        #   (b) If counts differ (prune occurred) → use global mean FDS
        #       without region_mask, since the mask references stale indices.
        fds = 0.0
        if self.pre_edit_features is not None:
            pre_n = self.pre_edit_features.shape[0]
            post_n = post_features.shape[0]
            if pre_n == post_n:
                # Same count → indices are aligned
                fds = feature_distribution_shift(
                    self.pre_edit_features.to(self.device),
                    post_features.to(self.device),
                    region_mask=edit_mask if edit_mask is not None else None,
                )
            else:
                # Prune occurred → global distribution comparison (no mask)
                fds = feature_distribution_shift(
                    self.pre_edit_features.to(self.device),
                    post_features.to(self.device),
                    region_mask=None,
                )

        # -- Adaptive finetune steps --
        rec_steps = compute_recommended_finetune_steps(scs, urp, fds)

        # -- Confidence: higher when we have more data points --
        n_views = len(scs_values)
        confidence = min(1.0, 0.3 + 0.2 * n_views + 0.1 * (1 if self.pre_edit_features is not None else 0))

        result = SemanticEvalResult(
            semantic_consistency_score=scs,
            unedited_preservation_score=urp,
            edit_precision=prec,
            edit_recall=rec,
            feature_distribution_shift=fds,
            recommended_finetune_steps=rec_steps,
            confidence=confidence,
            multi_view_consistency=mvc,
            per_view_scs=per_view_scs if per_view_scs else None,
            worst_view_idx=worst_view_idx,
            worst_view_scs=worst_view_scs,
        )

        logger.info("[SemanticEvaluator] Post-edit eval: %s", result.summary())
        return result

    def _project_mask_to_2d(
        self,
        edit_mask: Optional[torch.Tensor],
        batch: Dict[str, Any],
        H: int,
        W: int,
    ) -> torch.Tensor:
        """
        Project a 3D Gaussian edit mask onto a 2D image plane.

        Uses camera intrinsics/extrinsics from the batch to project the
        3D positions of edited Gaussians into pixel coordinates, then
        splats each projection with a small radius to form a binary mask.

        Falls back to a conservative center-crop if no camera data or
        no edit mask is available.

        Args:
            edit_mask: Boolean mask of edited Gaussians [N], or None.
            batch: Camera batch dict (must contain 'c2w' and 'fovx'/'fovy').
            H: Image height.
            W: Image width.

        Returns:
            Boolean 2D mask [H, W] on self.device.
        """
        edited_2d = torch.zeros(H, W, dtype=torch.bool, device=self.device)

        if edit_mask is None or edit_mask.sum() == 0:
            # No mask → conservative: mark center 50 % as edited
            h_s, w_s = H // 4, W // 4
            edited_2d[h_s:H - h_s, w_s:W - w_s] = True
            return edited_2d

        # Gather 3D positions of edited Gaussians
        xyz = self.system.geometry.get_xyz.detach()
        if edit_mask.shape[0] != xyz.shape[0]:
            # Mask length mismatch (e.g. after prune) → fallback
            h_s, w_s = H // 4, W // 4
            edited_2d[h_s:H - h_s, w_s:W - w_s] = True
            return edited_2d

        edited_xyz = xyz[edit_mask]  # [M, 3]

        c2w = batch.get("c2w")
        if c2w is None:
            h_s, w_s = H // 4, W // 4
            edited_2d[h_s:H - h_s, w_s:W - w_s] = True
            return edited_2d

        c2w_mat = c2w[0] if c2w.dim() == 3 else c2w  # [4, 4]
        w2c = torch.inverse(c2w_mat)  # [4, 4]

        fovx = batch.get("fovx", batch.get("fov", 0.8))
        fovy = batch.get("fovy", fovx)
        if isinstance(fovx, torch.Tensor):
            fovx = fovx.item()
        if isinstance(fovy, torch.Tensor):
            fovy = fovy.item()

        fx = W / (2.0 * math.tan(fovx / 2.0))
        fy = H / (2.0 * math.tan(fovy / 2.0))
        cx, cy = W / 2.0, H / 2.0

        # Project: world → camera → image
        pts_h = torch.cat([
            edited_xyz,
            torch.ones(edited_xyz.shape[0], 1, device=self.device),
        ], dim=1)  # [M, 4]
        pts_cam = (w2c @ pts_h.T).T[:, :3]  # [M, 3]

        valid = pts_cam[:, 2] > 0.01
        if valid.sum() == 0:
            h_s, w_s = H // 4, W // 4
            edited_2d[h_s:H - h_s, w_s:W - w_s] = True
            return edited_2d

        pts_cam = pts_cam[valid]
        u = (fx * pts_cam[:, 0] / pts_cam[:, 2] + cx).long()
        v = (fy * pts_cam[:, 1] / pts_cam[:, 2] + cy).long()

        splat_r = max(3, min(H, W) // 80)
        u_np = u.cpu().numpy()
        v_np = v.cpu().numpy()

        for ui, vi in zip(u_np, v_np):
            if 0 <= ui < W and 0 <= vi < H:
                r0 = max(0, vi - splat_r)
                r1 = min(H, vi + splat_r + 1)
                c0 = max(0, ui - splat_r)
                c1 = min(W, ui + splat_r + 1)
                edited_2d[r0:r1, c0:c1] = True

        return edited_2d
