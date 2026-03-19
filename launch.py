"""
Modules Launch Script
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="google.*")
import os
import sys
import argparse
import torch
import logging
from pathlib import Path

# Suppress verbose library logging
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('open_clip').setLevel(logging.ERROR)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.system.scene_lang import SceneLangSystem, SceneLangConfig
from modules.data import SceneDataModule, DataConfig
from modules.geometry.gaussian_base import GaussianModelConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Scene Editor Training and Testing")
    
    # Mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Train mode")
    group.add_argument("--save", action="store_true", help="Save visualization/interaction")
    group.add_argument("--edit", action="store_true", help="Edit scene")
    group.add_argument("--export", action="store_true", help="Export PLY")
    
    # Data
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--data_type", type=str, default="colmap", choices=["colmap", "nerf"], help="Data type")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    
    # Training
    parser.add_argument("--iterations", type=int, default=30000, help="Training iterations")
    parser.add_argument("--distill_lang_freq", type=int, default=800, help="Language distillation frequency")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    
    # Model
    parser.add_argument("--sh_degree", type=int, default=3, help="SH degree")
    parser.add_argument("--lang_feature_dim", type=int, default=3, help="Language feature dimension (3D required by CUDA rasterizer)")
    
    # Resume
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume")
    parser.add_argument("--ply", type=str, default=None, help="PLY path to load")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for interaction/editing")
    
    # Editing
    parser.add_argument("--edit_operation", type=str, default="translate", 
                        choices=["translate", "rotate", "scale", "delete"],
                        help="Editing operation")
    parser.add_argument("--edit_offset", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        help="Translation offset (x, y, z)")
    parser.add_argument("--edit_rotation", type=float, nargs=3, default=[0.0, 0.0, 15.0],
                        help="Rotation angles (roll, pitch, yaw in degrees)")
    parser.add_argument("--edit_scale", type=float, default=1.0,
                        help="Scale factor")
    parser.add_argument("--edit_threshold", type=float, default=0.5,
                        help="Selection threshold for editing")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID")
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mode = 'TRAIN' if args.train else 'SAVE' if args.save else 'EDIT' if args.edit else 'EXPORT'
    logger.info(f"Mode {mode}")
    
    # Initialize data module
    data_cfg = DataConfig(
        data_dir=args.data_dir,
        data_type=args.data_type,
        batch_size=args.batch_size,
    )
    data_module = SceneDataModule(data_cfg)
    
    # Initialize geometry config
    geom_cfg = GaussianModelConfig(
        sh_degree=args.sh_degree,
        lang_feature_dim=args.lang_feature_dim,
    )
    
    # Initialize system config
    system_cfg = SceneLangConfig(
        geometry=geom_cfg,
        distill_lang_freq=args.distill_lang_freq,
    )
    
    # Initialize system
    system = SceneLangSystem(system_cfg, device=args.device)
    
    # Initialize from point cloud
    if data_module.point_cloud is not None:
        system.create_from_pcd(data_module.point_cloud, spatial_lr_scale=1.0)
    
    # Load checkpoint or PLY
    if args.resume:
        system.load_checkpoint(args.resume)
    elif args.ply:
        system.load_ply(args.ply)
    
    # Validate language feature dimension after initialization
    actual_dim = system.geometry._language_feature.shape[1]
    expected_dim = args.lang_feature_dim
    if actual_dim != expected_dim:
        raise RuntimeError(
            f"CRITICAL ERROR: Language feature dimension mismatch!\n"
            f"Expected: {expected_dim}D (from --lang_feature_dim argument)\n"
            f"Actual:   {actual_dim}D (from loaded geometry)\n\n"
            f"This means you are loading an OLD checkpoint/PLY with {actual_dim}D features.\n\n"
            f"SOLUTION:\n"
            f"1. Delete old checkpoint: rm -rf {args.output_dir}\n"
            f"2. Remove --resume and --ply from your command\n"
            f"3. Start fresh training from point cloud\n"
        )
    
    if args.train:
        # Training mode
        system.training_setup()
        
        train_dataloader = data_module.train_dataloader()
        
        # Create persistent iterator for efficiency
        train_iter = iter(train_dataloader)
        
        logger.info(f"Training {args.iterations} iterations")
        
        from tqdm import tqdm
        pbar = tqdm(range(args.iterations), desc="Training")
        
        for iteration in pbar:
            # Get batch from persistent iterator
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                batch = next(train_iter)
            
            # Move to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(args.device)
            
            # Training step
            metrics = system.training_step(batch)
            
            # Update progress bar with loss and gaussians
            loss_val = metrics['loss'].item()
            n_gauss = metrics['num_gaussians']
            pbar.set_postfix({'loss': f'{loss_val:.4f}', 'gaussians': n_gauss})
            
            # Semantic feature distillation at step 100
            if (iteration + 1) == 100:
                logger.info(f"Starting semantic feature distillation...")
                # Use a fresh dataloader for distillation to cover all views
                # We need to make sure we iterate over the dataset
                distill_loader = data_module.train_dataset
                system.prepare_semantic_data(distill_loader)
                logger.info("Semantic feature distillation complete")

            # Save checkpoint only at step 15000
            if (iteration + 1) == 15000:
                ckpt_path = output_dir / f"checkpoint_{iteration+1}.pth"
                system.save_checkpoint(str(ckpt_path))
                ply_path = output_dir / f"point_cloud_{iteration+1}.ply"
                system.save_ply(str(ply_path))
        
        # Final save
        final_ckpt = output_dir / "final_checkpoint.pth"
        system.save_checkpoint(str(final_ckpt))
        
        final_ply = output_dir / "final_point_cloud.ply"
        system.save_ply(str(final_ply))
        
        logger.info("Training complete")
    
    elif args.save:
        # Save/Interaction mode
        from modules.utils.save import save_visualization
        
        logger.info("Save mode")
        save_visualization(
            system=system,
            data_module=data_module,
            output_dir=output_dir,
            prompt=args.prompt,
            device=args.device
        )
    
    elif args.edit:
        # Edit mode
        from modules.utils.edit import SceneEditor
        from modules.utils.edit_visualizer import render_editing_results, render_comparison
        
        logger.info("Edit mode")
        
        if not args.prompt:
            logger.error("--prompt is required for editing mode")
            return
        
        # Initialize editor
        editor = SceneEditor(system, device=args.device)
        
        # Backup original state for comparison
        editor.backup_parameters()
        
        # Prepare operation kwargs
        operation = args.edit_operation
        kwargs = {
            'threshold': args.edit_threshold,
        }
        
        if operation == 'translate':
            kwargs['offset'] = tuple(args.edit_offset)
        elif operation == 'rotate':
            kwargs['rotation'] = tuple(args.edit_rotation)
        elif operation == 'scale':
            kwargs['scale_factor'] = args.edit_scale
        
        # Execute editing
        result = editor.edit_scene(
            prompt=args.prompt,
            operation=operation,
            **kwargs
        )
        
        if not result['success']:
            logger.error(f"Editing failed: {result.get('message', 'Unknown error')}")
            return
        
        # Render edited scene
        logger.info("Rendering edited scene")
        video_path = render_editing_results(
            system=system,
            data_module=data_module,
            output_dir=output_dir,
            operation_name=f"{operation}_{args.prompt.replace(' ', '_')}",
            device=args.device
        )
        
        # Render comparison
        logger.info("Rendering comparison")
        comparison_path = render_comparison(
            system=system,
            data_module=data_module,
            output_dir=output_dir,
            operation_name=f"{operation}_{args.prompt.replace(' ', '_')}",
            editor=editor,
            device=args.device
        )
        
        # Save edited PLY
        edited_ply = output_dir / "save" / f"edited_{operation}_{args.prompt.replace(' ', '_')}.ply"
        system.save_ply(str(edited_ply))
        
        logger.info("Scene editing complete")
    
    elif args.export:
        # Export mode
        export_ply = output_dir / "exported_point_cloud.ply"
        system.save_ply(str(export_ply))
        logger.info(f"Exported PLY {export_ply}")


if __name__ == "__main__":
    main()
