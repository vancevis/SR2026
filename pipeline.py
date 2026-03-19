"""
COLMAP Data Pipeline
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required dependencies are installed"""
    dependencies = {
        'ffmpeg': 'ffmpeg -version',
        'colmap': 'colmap -h'
    }
    
    missing = []
    for name, cmd in dependencies.items():
        try:
            subprocess.run(
                cmd.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            logger.info(f"{name} is installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error(f"{name} is NOT installed")
            missing.append(name)
    
    if missing:
        logger.error(f"Missing dependencies: {', '.join(missing)}")
        logger.error("Installation instructions:")
        if 'ffmpeg' in missing:
            logger.error("  FFmpeg: https://ffmpeg.org/download.html")
            logger.error("    Ubuntu: sudo apt install ffmpeg")
            logger.error("    Windows: Download from https://ffmpeg.org/")
        if 'colmap' in missing:
            logger.error("  COLMAP: https://colmap.github.io/install.html")
            logger.error("    Ubuntu: sudo apt install colmap")
            logger.error("    Windows: Download from https://github.com/colmap/colmap/releases")
        return False
    
    return True


def extract_frames_from_video(video_path, output_dir, fps=1, quality=2):
    """
    Extract frames from video using ffmpeg
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        fps: Frames per second to extract (default: 1)
        quality: JPEG quality (1-31, lower is better, default: 2)
    
    Returns:
        bool: Success status
    """
    logger.info(f"Extracting frames from video: {video_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"FPS: {fps}, Quality: {quality}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Build ffmpeg command
    output_pattern = os.path.join(output_dir, "input_%04d.jpg")
    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-vf', f'fps={fps}',
        '-qscale:v', str(quality),
        output_pattern
    ]
    
    try:
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            return False
        
        # Count extracted frames
        num_frames = len([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
        logger.info(f"Extracted {num_frames} frames")
        return True
        
    except Exception as e:
        logger.error(f"Failed to extract frames: {e}")
        return False


def copy_images(source_dir, output_dir):
    """
    Copy images from source directory to output directory
    
    Args:
        source_dir: Source directory containing images
        output_dir: Destination directory
    
    Returns:
        bool: Success status
    """
    logger.info(f"Copying images from: {source_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Supported image extensions
    image_exts = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    
    # Find all images
    source_path = Path(source_dir)
    images = [f for f in source_path.iterdir() if f.suffix in image_exts]
    
    if not images:
        logger.error(f"No images found in {source_dir}")
        return False
    
    # Copy images
    for img in images:
        dst = os.path.join(output_dir, img.name)
        shutil.copy2(img, dst)
    
    logger.info(f"Copied {len(images)} images")
    return True


def run_colmap_reconstruction(workspace_dir, images_dir, quality='high', gpu=True):
    """
    Run COLMAP automatic reconstruction
    
    Args:
        workspace_dir: Workspace directory for COLMAP
        images_dir: Directory containing input images
        quality: Reconstruction quality ('low', 'medium', 'high', 'extreme')
        gpu: Whether to use GPU acceleration
    
    Returns:
        bool: Success status
    """
    logger.info("Starting COLMAP reconstruction")
    logger.info(f"Workspace: {workspace_dir}")
    logger.info(f"Images: {images_dir}")
    logger.info(f"Quality: {quality}")
    logger.info(f"GPU: {'Enabled' if gpu else 'Disabled'}")
    
    # Create workspace directory
    os.makedirs(workspace_dir, exist_ok=True)
    
    # Build COLMAP command
    cmd = [
        'colmap', 'automatic_reconstructor',
        '--image_path', str(images_dir),
        '--workspace_path', str(workspace_dir),
        '--quality', quality
    ]
    
    if not gpu:
        cmd.extend(['--use_gpu', '0'])
    
    try:
        logger.info(f"Running: {' '.join(cmd)}")
        logger.info("This may take several minutes depending on the number of images...")
        
        # Run COLMAP with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output
        for line in process.stdout:
            line = line.strip()
            if line:
                logger.info(f"[COLMAP] {line}")
        
        process.wait()
        
        if process.returncode != 0:
            logger.error("COLMAP reconstruction failed")
            return False
        
        # Check for output directory
        sparse_dir = os.path.join(workspace_dir, 'sparse', '0')
        if not os.path.exists(sparse_dir):
            logger.error(f"Sparse reconstruction directory not found: {sparse_dir}")
            return False

        # Check for binary files and convert to text if necessary
        bin_files = ['cameras.bin', 'images.bin', 'points3D.bin']
        txt_files = ['cameras.txt', 'images.txt', 'points3D.txt']
        
        has_bin = all(os.path.exists(os.path.join(sparse_dir, f)) for f in bin_files)
        has_txt = all(os.path.exists(os.path.join(sparse_dir, f)) for f in txt_files)
        
        if has_bin and not has_txt:
            logger.info("COLMAP produced binary files. Converting to text...")
            convert_cmd = [
                'colmap', 'model_converter',
                '--input_path', str(sparse_dir),
                '--output_path', str(sparse_dir),
                '--output_type', 'TXT'
            ]
            
            try:
                convert_result = subprocess.run(
                    convert_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True
                )
                logger.info("Conversion successful")
            except subprocess.CalledProcessError as e:
                logger.error(f"Model conversion failed: {e.stderr}")
                return False
        
        # Verify output
        required_files = ['cameras.txt', 'images.txt', 'points3D.txt']
        
        for fname in required_files:
            fpath = os.path.join(sparse_dir, fname)
            if not os.path.exists(fpath):
                logger.error(f"Missing required file: {fname}")
                # List directory contents for debugging
                if os.path.exists(sparse_dir):
                    logger.info(f"Contents of {sparse_dir}:")
                    for f in os.listdir(sparse_dir):
                        logger.info(f"  - {f}")
                else:
                    logger.error(f"Directory {sparse_dir} does not exist")
                return False
        
        logger.info("COLMAP reconstruction completed successfully")
        logger.info(f"Output: {sparse_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to run COLMAP: {e}")
        return False


def verify_colmap_output(workspace_dir):
    """
    Verify COLMAP output structure
    
    Args:
        workspace_dir: COLMAP workspace directory
    
    Returns:
        dict: Statistics about the reconstruction
    """
    sparse_dir = os.path.join(workspace_dir, 'sparse', '0')
    
    stats = {
        'valid': False,
        'num_cameras': 0,
        'num_images': 0,
        'num_points': 0
    }
    
    try:
        # Count cameras
        cameras_file = os.path.join(sparse_dir, 'cameras.txt')
        if os.path.exists(cameras_file):
            with open(cameras_file, 'r') as f:
                stats['num_cameras'] = sum(1 for line in f if not line.startswith('#') and line.strip())
        
        # Count images
        images_file = os.path.join(sparse_dir, 'images.txt')
        if os.path.exists(images_file):
            with open(images_file, 'r') as f:
                lines = [line for line in f if not line.startswith('#') and line.strip()]
                stats['num_images'] = len(lines) // 2  # Each image has 2 lines
        
        # Count points
        points_file = os.path.join(sparse_dir, 'points3D.txt')
        if os.path.exists(points_file):
            with open(points_file, 'r') as f:
                stats['num_points'] = sum(1 for line in f if not line.startswith('#') and line.strip())
        
        stats['valid'] = stats['num_cameras'] > 0 and stats['num_images'] > 0 and stats['num_points'] > 0
        
    except Exception as e:
        logger.error(f"Failed to verify output: {e}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='COLMAP Data Pipeline - Convert video or images to COLMAP format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a video
  python pipeline.py --input video.mp4 --output data/my_scene --fps 1

  # Process images in a folder
  python pipeline.py --input images/ --output data/my_scene
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input video file or directory containing images'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for COLMAP data'
    )
    
    parser.add_argument(
        '--fps',
        type=float,
        default=1.0,
        help='Frames per second to extract from video (default: 1.0)'
    )
    
    parser.add_argument(
        '--jpeg-quality',
        type=int,
        default=2,
        choices=range(1, 32),
        metavar='[1-31]',
        help='JPEG quality for extracted frames (1=best, 31=worst, default: 2)'
    )
    
    parser.add_argument(
        '--quality',
        type=str,
        default='high',
        choices=['low', 'medium', 'high', 'extreme'],
        help='COLMAP reconstruction quality (default: high)'
    )
    
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU acceleration for COLMAP'
    )
    
    parser.add_argument(
        '--skip-colmap',
        action='store_true',
        help='Skip COLMAP reconstruction (only extract frames/copy images)'
    )
    
    args = parser.parse_args()
    
    # Print header
    logger.info("COLMAP Data Pipeline for Scene Editor")
    
    # Check dependencies
    if not check_dependencies():
        logger.error("\nPlease install missing dependencies and try again")
        return 1
    
    # Determine input type
    input_path = Path(args.input)
    
    if not input_path.exists():
        logger.error(f"Input path does not exist: {args.input}")
        return 1
    
    # Create output directory structure
    output_dir = Path(args.output)
    images_dir = output_dir / 'images'
    
    # Process input
    is_video = input_path.is_file() and input_path.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.flv'}
    
    if is_video:
        logger.info(f"Input type: Video ({input_path.suffix})")
        if not extract_frames_from_video(
            input_path,
            images_dir,
            fps=args.fps,
            quality=args.jpeg_quality
        ):
            logger.error("Failed to extract frames from video")
            return 1
    
    elif input_path.is_dir():
        logger.info("Input type: Image directory")
        if not copy_images(input_path, images_dir):
            logger.error("Failed to copy images")
            return 1
    
    else:
        logger.error(f"Unsupported input type: {args.input}")
        logger.error("Input must be a video file (.mp4, .avi, .mov, .mkv, .flv) or a directory containing images")
        return 1
    
    # Skip COLMAP if requested
    if args.skip_colmap:
        logger.info("Skipping COLMAP reconstruction (--skip-colmap)")
        logger.info(f"Images saved to: {images_dir}")
        return 0
    
    # Run COLMAP reconstruction
    if not run_colmap_reconstruction(
        output_dir,
        images_dir,
        quality=args.quality,
        gpu=not args.no_gpu
    ):
        logger.error("COLMAP reconstruction failed")
        return 1
    
    # Verify output
    logger.info("Verifying reconstruction...")
    
    stats = verify_colmap_output(output_dir)
    
    if stats['valid']:
        logger.info("Reconstruction verified successfully")
        logger.info(f"  Cameras: {stats['num_cameras']}")
        logger.info(f"  Images: {stats['num_images']}")
        logger.info(f"  3D Points: {stats['num_points']}")
        logger.info("Pipeline completed successfully")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"COLMAP data: {output_dir / 'sparse' / '0'}")
        logger.info("You can now use this data with Scene Editor:")
        logger.info(f"  python launch.py --data_dir {output_dir}")
        return 0
    else:
        logger.error("Reconstruction verification failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
