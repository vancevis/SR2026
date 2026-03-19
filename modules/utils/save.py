"""
Semantic Field PCA Visualization with LLM Interaction
"""

import torch
import numpy as np
import imageio
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sklearn.decomposition import PCA

try:
    from .llm import SceneLLM
    LLM_AVAILABLE = True
except Exception:
    LLM_AVAILABLE = False

def save_visualization(
    system,
    data_module,
    output_dir: Path,
    prompt: str = None,
    device: str = "cuda"
):
    """
    Save semantic field visualization using PCA decomposition
    Achieves clear object segmentation like reference image
    
    Args:
        system: SceneLangSystem with trained semantic features
        data_module: SceneDataModule
        output_dir: Output directory
        prompt: Optional text prompt for LLM interaction
        device: Device to run on
    """
    save_dir = output_dir / "save"
    save_dir.mkdir(parents=True, exist_ok=True)
    dataloader = data_module.test_dataloader()
    
    #print("Phase 1: Accumulating semantic features for PCA")
    
    # Storage for PCA computation
    rendered_langs = []
    rendered_rgbs = []
    
    # First pass: collect all semantic features and RGB
    for batch in tqdm(dataloader, desc="Accumulating", ncols=80):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        
        with torch.no_grad():
            output = system(batch)
            lang_compressed = output['lang'][0]  # [H, W, 3]
            rendered_langs.append(lang_compressed)
            rendered_rgbs.append(output['comp_rgb'][0].cpu())
    
    #print("Phase 2: Computing global PCA transform")
    
    # Sample pixels for PCA (avoid memory overflow)
    sample_pixels = []
    for lang in rendered_langs[::5]:  # every 5th frame
        flat = lang.reshape(-1, 3)
        indices = torch.randperm(flat.shape[0])[:1000]
        sample_pixels.append(flat[indices])
    
    samples_3d = torch.cat(sample_pixels, dim=0).cpu().numpy()
    
    # Check if features are meaningful (not all zeros or very small)
    feature_magnitude = np.abs(samples_3d).mean()
    feature_std = np.std(samples_3d, axis=0)
    print(f"Feature magnitude: {feature_magnitude:.6f}")
    print(f"Feature std: [{feature_std[0]:.6f}, {feature_std[1]:.6f}, {feature_std[2]:.6f}]")
    
    if feature_magnitude < 0.001:
        print(f"WARNING: Language features are nearly zero")
        print("This likely means semantic distillation did not run during training.")
        print("Please train for at least 100+ steps to trigger semantic feature learning.")
    
    # CRITICAL: Apply PCA directly on 3D rendered features
    # The 3D bottleneck has already compressed semantics - decoding to 512D doesn't add information
    # This matches the approach: render 3D -> PCA to RGB (similar to 3DitScene)
    pca = PCA(n_components=3)
    pca.fit(samples_3d)
    variance = pca.explained_variance_ratio_
    print(f"PCA variance: [{variance[0]:.3f}, {variance[1]:.3f}, {variance[2]:.3f}]")
    print(f"Total variance explained: {variance.sum():.3f}")
    
    #print("Phase 3: Generating visualizations")
    
    rgb_frames = []
    pca_frames = []
    
    # Second pass: apply PCA and save (uses cached RGB from first pass)
    for idx, (out_rgb, lang_3d) in enumerate(zip(rendered_rgbs, rendered_langs)):
        rgb_img = (out_rgb.clamp(0, 1).numpy() * 255).astype(np.uint8)
        Image.fromarray(rgb_img).save(save_dir / f"rgb_{idx:05d}.png")
        rgb_frames.append(rgb_img)
        
        # Process semantic features
        H, W, _ = lang_3d.shape
        flat_3d = lang_3d.reshape(-1, 3).cpu().numpy()
        
        # PCA transform: 3D -> 3D RGB
        # Apply PCA directly on rendered 3D features (no decode needed)
        flat_pca = pca.transform(flat_3d)
        
        # Robust normalization using percentiles
        low = np.percentile(flat_pca, 1, axis=0)
        high = np.percentile(flat_pca, 99, axis=0)
        flat_pca = (flat_pca - low) / (high - low + 1e-9)
        flat_pca = np.clip(flat_pca, 0, 1)
        
        pca_img = (flat_pca.reshape(H, W, 3) * 255).astype(np.uint8)
        Image.fromarray(pca_img).save(save_dir / f"action_{idx:05d}.png")
        pca_frames.append(pca_img)
    
    # Save videos
    imageio.mimwrite(save_dir / "video_rgb.mp4", rgb_frames, fps=30, quality=8)
    imageio.mimwrite(save_dir / "video_action.mp4", pca_frames, fps=30, quality=8)
    
    # LLM interaction
    if prompt and LLM_AVAILABLE:
        print(f"LLM Query: {prompt}")
        llm = SceneLLM()
        if llm.model:
            response = llm.chat_with_scene(
                query=prompt,
                image_paths=[save_dir / "rgb_00000.png", save_dir / "action_00000.png"]
            )
            print(f"LLM Response: {response}")
            with open(save_dir / "llm_response.txt", "w") as f:
                f.write(f"Q: {prompt}\n\nA: {response}")
    
    print(f"complete")