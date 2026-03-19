"""
Edit Visualization Module
Renders and saves editing results
"""

import os
import torch
import numpy as np
import imageio
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def render_editing_results(
    system,
    data_module,
    output_dir: Path,
    operation_name: str = "edited",
    device: str = "cuda"
):
    """
    Render and save editing results (images and video)
    
    Args:
        system: SceneLangSystem with edited Gaussians
        data_module: SceneDataModule for camera views
        output_dir: Output directory
        operation_name: Name of the operation (for file naming)
        device: Device to run on
    """
    logger.info(f"Rendering editing results: {operation_name}")
    
    # Create save directory
    save_dir = output_dir / "save" / f"edit_{operation_name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get cameras
    dataloader = data_module.test_dataloader()
    
    # Setup video writer
    fps = 30
    rgb_frames = []
    
    # Render frames
    logger.info(f"[EditVisualizer] Rendering {len(dataloader)} views...")
    for idx, batch in enumerate(tqdm(dataloader, desc="Rendering edited scene")):
        # Move to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        
        # Render
        with torch.no_grad():
            output = system(batch)
        
        # RGB Image
        rgb = output['comp_rgb'][0].clamp(0, 1).cpu().numpy()
        rgb_img = (rgb * 255).astype(np.uint8)
        rgb_pil = Image.fromarray(rgb_img)
        rgb_pil.save(save_dir / f"rgb_{idx:05d}.png")
        rgb_frames.append(rgb_img)
    
    # Save Video
    logger.info(f"[EditVisualizer] Saving video...")
    video_path = save_dir / f"video_{operation_name}.mp4"
    imageio.mimwrite(video_path, rgb_frames, fps=fps, quality=8)
    
    logger.info(f"Results saved to {save_dir}")
    logger.info(f"Video saved: {video_path}")
    
    return str(video_path)


def render_comparison(
    system,
    data_module,
    output_dir: Path,
    operation_name: str,
    editor,
    device: str = "cuda"
):
    """
    Render before/after comparison
    
    Args:
        system: SceneLangSystem
        data_module: SceneDataModule
        output_dir: Output directory
        operation_name: Name of operation
        editor: SceneEditor with backup
        device: Device to run on
    """
    logger.info("[EditVisualizer] Rendering before/after comparison...")
    
    # Create comparison directory
    comp_dir = output_dir / "save" / f"comparison_{operation_name}"
    comp_dir.mkdir(parents=True, exist_ok=True)
    
    # Get a sample view
    dataloader = data_module.test_dataloader()
    sample_batch = next(iter(dataloader))
    for k, v in sample_batch.items():
        if isinstance(v, torch.Tensor):
            sample_batch[k] = v.to(device)
    
    # Render AFTER
    with torch.no_grad():
        output_after = system(sample_batch)
    rgb_after = output_after['comp_rgb'][0].cpu().numpy()
    rgb_after_img = (rgb_after * 255).astype(np.uint8)
    
    # Restore and render BEFORE
    editor.restore_parameters()
    with torch.no_grad():
        output_before = system(sample_batch)
    rgb_before = output_before['comp_rgb'][0].cpu().numpy()
    rgb_before_img = (rgb_before * 255).astype(np.uint8)
    
    # Save comparison
    Image.fromarray(rgb_before_img).save(comp_dir / "before.png")
    Image.fromarray(rgb_after_img).save(comp_dir / "after.png")
    
    # Create side-by-side comparison
    H, W = rgb_before_img.shape[:2]
    comparison = np.zeros((H, W*2, 3), dtype=np.uint8)
    comparison[:, :W, :] = rgb_before_img
    comparison[:, W:, :] = rgb_after_img
    Image.fromarray(comparison).save(comp_dir / "comparison.png")
    
    logger.info(f"[EditVisualizer] Comparison saved to {comp_dir}")
    
    return str(comp_dir / "comparison.png")
