"""
SAM + CLIP Feature Extractor
"""

import torch
import torch.nn as nn
import torchvision
import numpy as np
import cv2
import os
import sys
import logging
from dataclasses import dataclass
from typing import Tuple, List, Optional

# Suppress verbose logging from libraries
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False

try:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

try:
    from mobile_sam import sam_model_registry as mobile_sam_registry
    from mobile_sam import SamAutomaticMaskGenerator as MobileSamMaskGenerator
    MOBILE_SAM_AVAILABLE = True
except ImportError:
    MOBILE_SAM_AVAILABLE = False


@dataclass
class SamClipConfig:
    """Configuration for SAM+CLIP"""
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    sam_ckpt_path: str = "ckpts/sam_vit_h_4b8939.pth"
    mobile_sam_ckpt_path: str = "ckpts/mobile_sam.pt"
    use_mobile_sam: bool = True
    feature_level: int = 3
    vis_pca_feature: bool = True
    negatives: Tuple[str, ...] = ("object", "things", "stuff", "texture")
    positives: Tuple[str, ...] = ("",)


class OpenCLIPNetwork(nn.Module):
    """OpenCLIP network wrapper"""
    
    def __init__(self, cfg: SamClipConfig):
        super().__init__()
        self.cfg = cfg
        
        # Image preprocessing
        self.process = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
        
        # Load CLIP model (suppress verbose output)
        import contextlib
        with open(os.devnull, 'w') as _devnull:
            with contextlib.redirect_stdout(_devnull), \
                 contextlib.redirect_stderr(_devnull):
                model, _, _ = open_clip.create_model_and_transforms(
                    cfg.clip_model_type,
                    pretrained=cfg.clip_model_pretrained,
                    precision="fp16",
                )
        model.eval()
        self.model = model.to("cuda")
        self.tokenizer = open_clip.get_tokenizer(cfg.clip_model_type)
        self.clip_n_dims = cfg.clip_n_dims
        
        # Positive and negative prompts
        self.positives = cfg.positives
        self.negatives = cfg.negatives
        
        # Pre-compute embeddings
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases)
        
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)
    
    def set_positives(self, text_list: List[str]):
        """Update positive prompts"""
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
    
    def encode_image(self, input: torch.Tensor) -> torch.Tensor:
        """Encode images to CLIP embeddings"""
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)
    
    def get_relevancy(self, embed: torch.Tensor, positive_id: int = 0) -> torch.Tensor:
        """Compute relevancy scores"""
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)
        output = torch.mm(embed, p.T)
        
        positive_vals = output[..., positive_id:positive_id + 1]
        negative_vals = output[..., len(self.positives):]
        repeated_pos = positive_vals.repeat(1, len(self.negatives))
        
        sims = torch.stack((repeated_pos, negative_vals), dim=-1)
        softmax = torch.softmax(10 * sims, dim=-1)
        best_id = softmax[..., 0].argmin(dim=1)
        
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[:, 0, :]


def get_seg_img(mask: dict, image: np.ndarray) -> np.ndarray:
    """Extract segmented region from mask"""
    image = image.copy()
    image[mask['segmentation'] == 0] = np.array([0, 0, 0], dtype=np.uint8)
    x, y, w, h = np.int32(mask['bbox'])
    seg_img = image[y:y+h, x:x+w, ...]
    return seg_img


def pad_img(img: np.ndarray) -> np.ndarray:
    """Pad image to square"""
    h, w, _ = img.shape
    l = max(w, h)
    pad = np.zeros((l, l, 3), dtype=np.uint8)
    if h > w:
        pad[:, (h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad


def mask_nms(masks: torch.Tensor, scores: torch.Tensor,
             iou_thr: float = 0.7, score_thr: float = 0.1,
             inner_thr: float = 0.2) -> torch.Tensor:
    """Mask non-maximum suppression"""
    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    
    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)
    
    # Vectorized IoU and inner-IoU computation
    flat = masks_ord.reshape(num_masks, -1).float()  # [N, H*W]
    intersection_matrix = flat @ flat.T  # [N, N]
    union_matrix = masks_area.unsqueeze(0) + masks_area.unsqueeze(1) - intersection_matrix
    iou_matrix = intersection_matrix / (union_matrix + 1e-8)

    # Inner IoU: ratio of intersection to each mask's area
    ratio_i = intersection_matrix / (masks_area.unsqueeze(1) + 1e-8)  # intersection / area[i]
    ratio_j = intersection_matrix / (masks_area.unsqueeze(0) + 1e-8)  # intersection / area[j]
    inner_iou_vals = 1.0 - ratio_j * ratio_i

    inner_iou_matrix = torch.zeros_like(iou_matrix)
    # Case 1: ratio_i < 0.5 and ratio_j >= 0.85 → store in [i, j]
    cond1 = (ratio_i < 0.5) & (ratio_j >= 0.85)
    inner_iou_matrix[cond1] = inner_iou_vals[cond1]
    # Case 2: ratio_i >= 0.85 and ratio_j < 0.5 → store in [j, i]
    cond2 = (ratio_i >= 0.85) & (ratio_j < 0.5)
    inner_iou_matrix.T[cond2] = inner_iou_vals[cond2]
    
    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)
    
    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr
    
    if keep_conf.sum() == 0:
        index = scores.topk(min(3, len(scores))).indices
        keep_conf[index] = True
    
    keep = keep & keep_conf & keep_inner_u & keep_inner_l
    return idx[keep]


def masks_update(masks_lvl: List[dict], **kwargs) -> List[dict]:
    """Update masks using NMS"""
    if len(masks_lvl) == 0:
        return masks_lvl
    
    seg_pred = torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
    iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
    stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))
    
    scores = stability * iou_pred
    keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
    
    keep = keep_mask_nms.int().cpu().numpy()
    return [m for i, m in enumerate(masks_lvl) if i in keep]


class SamClip(nn.Module):
    """
    SAM + CLIP Feature Extractor
    Complete implementation based on 3DitScene
    """
    
    def __init__(self, cfg: SamClipConfig = None):
        super().__init__()
        self.cfg = cfg or SamClipConfig()
        
        if not OPEN_CLIP_AVAILABLE:
            raise ImportError("open_clip is required")
        
        # Initialize CLIP
        self.model = OpenCLIPNetwork(self.cfg)
        self.clip_n_dims = self.cfg.clip_n_dims
        self.tokenizer = open_clip.get_tokenizer(self.cfg.clip_model_type)
        
        # Initialize SAM
        self._init_sam()
        
    
    def _init_sam(self):
        """Initialize SAM model"""
        if self.cfg.use_mobile_sam and MOBILE_SAM_AVAILABLE:
            import os
            if os.path.exists(self.cfg.mobile_sam_ckpt_path):
                mobile_sam = mobile_sam_registry["vit_t"](checkpoint=self.cfg.mobile_sam_ckpt_path)
                mobile_sam.to(device="cuda")
                mobile_sam.eval()
                self.mask_generator = MobileSamMaskGenerator(mobile_sam)
                return
        
        if SAM_AVAILABLE:
            import os
            if os.path.exists(self.cfg.sam_ckpt_path):
                sam = sam_model_registry["vit_h"](checkpoint=self.cfg.sam_ckpt_path).to("cuda")
                self.mask_generator = SamAutomaticMaskGenerator(
                    model=sam,
                    points_per_side=32,
                    points_per_batch=64,
                    pred_iou_thresh=0.7,
                    box_nms_thresh=0.7,
                    stability_score_thresh=0.85,
                    crop_n_layers=1,
                    crop_n_points_downscale_factor=1,
                    min_mask_region_area=100,
                )
            else:
                raise RuntimeError(f"SAM checkpoint not found: {self.cfg.sam_ckpt_path}")
        else:
            raise RuntimeError("No SAM model available")
    
    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract SAM+CLIP features
        
        Args:
            image: [1, 3, H, W] uint8 tensor (0-255)
        
        Returns:
            embeddings: [N, 512] CLIP embeddings
            seg_map: [H, W] segment index map
            mask: [H, W] valid region mask
        """
        # Convert to numpy
        img_np = image.detach().cpu()
        # Input is RGB (from PIL), so just convert to numpy HWC
        img_np = img_np[0].permute(1, 2, 0).numpy().astype(np.uint8)
        H, W = img_np.shape[:2]
        
        # Generate masks
        masks = self.mask_generator.generate(img_np)
        
        # Post-process with NMS
        masks = masks_update(masks, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)
        
        # Sort masks by area descending (so small masks overwrite large ones)
        # Note: We paint largest first, so smallest will be painted last and stay on top
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        if len(masks) == 0:
            return (
                torch.zeros(1, self.clip_n_dims, device="cuda"),
                torch.full((H, W), -1, dtype=torch.long, device="cuda"),
                torch.zeros(H, W, dtype=torch.bool, device="cuda")
            )
        
        # Extract segment images
        seg_img_list = []
        seg_map = -np.ones((H, W), dtype=np.int32)
        
        for i, mask in enumerate(masks):
            seg_img = get_seg_img(mask, img_np)
            pad_seg_img = cv2.resize(pad_img(seg_img), (224, 224))
            seg_img_list.append(pad_seg_img)
            seg_map[mask['segmentation']] = i
        
        # Stack and convert to tensor
        seg_imgs = np.stack(seg_img_list, axis=0)
        seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0, 3, 1, 2) / 255.0).to("cuda")
        
        # Compute CLIP embeddings
        with torch.no_grad():
            clip_embed = self.model.encode_image(seg_imgs)
            clip_embed = clip_embed / clip_embed.norm(dim=-1, keepdim=True)
        
        seg_map_tensor = torch.from_numpy(seg_map).long().to("cuda")
        mask_tensor = (seg_map_tensor != -1)
        
        return clip_embed.float(), seg_map_tensor, mask_tensor
