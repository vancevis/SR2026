"""
Background Inpainting for 3D Gaussian Splatting
"""

import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
import logging

try:
    from diffusers import StableDiffusionInpaintPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logging.warning("diffusers not installed. Install with: pip install diffusers transformers accelerate")


class BackgroundInpainter:
    """
    Handles background inpainting using Stable Diffusion
    Based on 3DitScene's implementation
    """
    
    def __init__(self, model_id="runwayml/stable-diffusion-inpainting"):
        """
        Initialize inpainting pipeline
        
        Args:
            model_id: HuggingFace model ID for SD inpainting
        """
        self.model_id = model_id
        self.pipe = None
        self.enabled = False
        
    def initialize(self):
        """Load the inpainting model"""
        if not DIFFUSERS_AVAILABLE:
            logging.error("Cannot initialize inpainting: diffusers not installed")
            return False
        
        try:
            logging.info(f"Loading SD Inpainting model: {self.model_id}...")
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                safety_checker=None  # Disable for speed
            )
            self.enabled = True
            logging.info("SD Inpainting model loaded successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to load SD Inpainting: {e}")
            return False
    
    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str = "natural background, photorealistic, high quality",
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50
    ) -> Image.Image:
        """
        Inpaint masked region using Stable Diffusion
        
        Args:
            image: PIL Image (RGB)
            mask: PIL Image (L or RGB, white=inpaint region)
            prompt: Text prompt describing the background
            guidance_scale: CFG scale
            num_inference_steps: Number of denoising steps
            
        Returns:
            Inpainted PIL Image
        """
        if not self.enabled:
            logging.warning("Inpainting not enabled, returning original image")
            return image
        
        # Prepare inputs
        original_size = image.size
        img_512 = image.convert("RGB").resize((512, 512), Image.LANCZOS)
        mask_512 = mask.convert("L").resize((512, 512), Image.LANCZOS)
        
        # Move to GPU
        self.pipe.to("cuda")
        
        try:
            # Run inpainting
            logging.info(f"Running SD inpainting: '{prompt}'")
            result = self.pipe(
                prompt=prompt,
                image=img_512,
                mask_image=mask_512,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                height=512,
                width=512
            ).images[0]
            
            # Resize back
            result = result.resize(original_size, Image.LANCZOS)
            
        finally:
            # Clean up GPU memory
            self.pipe.to("cpu")
            torch.cuda.empty_cache()
        
        return result
    
    def __del__(self):
        """Clean up resources"""
        if self.pipe is not None:
            del self.pipe
            torch.cuda.empty_cache()


def dilate_mask(mask: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """
    Dilate binary mask to expand inpainting region
    
    Args:
        mask: Binary mask [H, W]
        kernel_size: Dilation kernel size
        
    Returns:
        Dilated mask [H, W]
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    return dilated > 0


def blur_mask_edges(mask: np.ndarray, kernel_size: int = 21) -> np.ndarray:
    """
    Blur mask edges for smooth blending
    
    Args:
        mask: Binary mask [H, W]
        kernel_size: Blur kernel size
        
    Returns:
        Blurred mask [H, W] with smooth edges
    """
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)
    return blurred


def create_inpaint_mask_from_gaussians(
    gaussian_mask: torch.Tensor,
    gaussian_xyz: torch.Tensor,
    c2w: torch.Tensor,
    fovx: float,
    fovy: float,
    height: int,
    width: int,
    dilate_kernel: int = 11,
    splat_radius: int = 3,
) -> Image.Image:
    """
    Create inpainting mask from Gaussian selection mask by projecting
    selected 3D Gaussian positions to 2D image coordinates.
    
    Args:
        gaussian_mask: Boolean mask for selected Gaussians [N]
        gaussian_xyz: 3D positions of ALL Gaussians [N, 3]
        c2w: Camera-to-world transform [4, 4]
        fovx: Horizontal field of view (radians)
        fovy: Vertical field of view (radians)
        height: Image height
        width: Image width
        dilate_kernel: Kernel size for dilation
        splat_radius: Pixel radius for each projected point
        
    Returns:
        PIL Image mask (white=inpaint)
    """
    import math

    mask_2d = np.zeros((height, width), dtype=np.uint8)

    selected_xyz = gaussian_xyz[gaussian_mask]  # [M, 3]
    if selected_xyz.shape[0] == 0:
        return Image.fromarray(mask_2d, mode='L')

    device = gaussian_xyz.device
    w2c = torch.inverse(c2w.to(device))

    fx = width / (2.0 * math.tan(fovx / 2.0))
    fy = height / (2.0 * math.tan(fovy / 2.0))
    cx, cy = width / 2.0, height / 2.0

    pts_h = torch.cat([
        selected_xyz,
        torch.ones(selected_xyz.shape[0], 1, device=device),
    ], dim=1)  # [M, 4]
    pts_cam = (w2c @ pts_h.T).T[:, :3]  # [M, 3]

    valid = pts_cam[:, 2] > 0.01
    if valid.sum() > 0:
        pts_cam = pts_cam[valid]
        u = (fx * pts_cam[:, 0] / pts_cam[:, 2] + cx).long().cpu().numpy()
        v = (fy * pts_cam[:, 1] / pts_cam[:, 2] + cy).long().cpu().numpy()

        for ui, vi in zip(u, v):
            if 0 <= ui < width and 0 <= vi < height:
                r0 = max(0, vi - splat_radius)
                r1 = min(height, vi + splat_radius + 1)
                c0 = max(0, ui - splat_radius)
                c1 = min(width, ui + splat_radius + 1)
                mask_2d[r0:r1, c0:c1] = 255
    
    # Dilate to expand coverage
    mask_dilated = dilate_mask(mask_2d, dilate_kernel)
    
    # Convert to PIL
    mask_img = Image.fromarray((mask_dilated * 255).astype(np.uint8), mode='L')
    
    return mask_img


def image_to_pointcloud_world(
    image: Image.Image,
    mask: Image.Image,
    c2w: torch.Tensor,
    depth_estimate: float = 5.0,
    fov: float = 60.0,
    device: str = "cuda"
) -> tuple:
    """
    Convert image pixels to 3D points in world space (corrected)
    
    Args:
        image: PIL Image (RGB)
        mask: PIL Image (white=pixels to convert)
        c2w: Camera to world transformation matrix [4, 4]
        depth_estimate: Estimated depth for all pixels
        fov: Field of view in degrees
        device: Device to place tensors on
        
    Returns:
        points: [N, 3] numpy array or torch tensor in world space
        colors: [N, 3] numpy array or torch tensor in [0, 1]
    """
    # Convert to arrays
    img_arr = np.array(image) / 255.0  # [H, W, 3]
    mask_arr = np.array(mask.convert('L')) > 128  # [H, W]
    
    h, w = mask_arr.shape
    
    # Get valid pixels
    ys, xs = np.where(mask_arr)
    
    if len(xs) == 0:
        return torch.zeros((0, 3), device=device), torch.zeros((0, 3), device=device)
    
    # Compute focal length from FOV
    focal_length = (w / 2) / np.tan(np.radians(fov / 2))
    
    # Compute 3D positions in camera space (assuming centered camera)
    cx, cy = w / 2, h / 2
    x_cam = (xs - cx) * depth_estimate / focal_length
    y_cam = (ys - cy) * depth_estimate / focal_length
    z_cam = np.full_like(x_cam, depth_estimate)
    
    # Convert to homogenous coordinates
    pts_cam = torch.tensor(np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=1), 
                           dtype=torch.float32, device=device)  # [N, 4]
    
    # Get colors
    colors = torch.tensor(img_arr[ys, xs], dtype=torch.float32, device=device)  # [N, 3]
    
    # Transform to world space using c2w
    c2w_device = c2w.to(device).float()
    pts_world = (c2w_device @ pts_cam.T).T[:, :3]  # [N, 3]
    
    return pts_world.cpu().numpy(), colors.cpu().numpy()


def merge_inpainted_background(
    gaussian_model,
    inpainted_image: Image.Image,
    mask: Image.Image,
    camera_params: dict,
    depth_estimate: float = 5.0
):
    """
    Merge inpainted background into Gaussian model
    
    Args:
        gaussian_model: GaussianBaseModel instance
        inpainted_image: Inpainted PIL Image
        mask: Mask indicating inpainted region
        camera_params: Camera parameters dict (must include c2w)
        depth_estimate: Depth estimate for new points
    """
    # Convert inpainted pixels to 3D points
    c2w = camera_params.get('c2w')
    if c2w is None:
        c2w = torch.eye(4)
        logging.warning("No c2w matrix provided, using identity matrix")
    elif c2w.dim() == 3:
        c2w = c2w[0]
        
    points, colors = image_to_pointcloud_world(
        inpainted_image,
        mask,
        c2w=c2w,
        depth_estimate=depth_estimate,
        fov=camera_params.get('fov', 60.0),
        device=gaussian_model.get_xyz.device
    )
    
    if points.shape[0] == 0:
        logging.warning("No background points generated")
        return
    
    # Sample points to avoid too many (optional)
    max_points = 10000
    if points.shape[0] > max_points:
        indices = np.random.choice(points.shape[0], max_points, replace=False)
        points = points[indices]
        colors = colors[indices]
    
    logging.info(f"Adding {points.shape[0]} background points to scene")
    
    # Use the merge method from gaussian_model
    if hasattr(gaussian_model, 'merge_background_points'):
        gaussian_model.merge_background_points(points, colors)
    else:
        logging.error("GaussianBaseModel doesn't have merge_background_points method")
