import os
import argparse
import time
import torch
from pathlib import Path
from PIL import Image
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Batch process multiview folders to 3D models.")
    parser.add_argument("--input_dir", type=str, required=True, help="Root directory containing image folders.")
    parser.add_argument("--output_dir", type=str, default="output", help="Root directory for outputs.")
    parser.add_argument("--shape_model", type=str, default="tencent/Hunyuan3D-2mv", help="Path to shape generation model.")
    parser.add_argument("--shape_subfolder", type=str, default="hunyuan3d-dit-v2-mv", help="Subfolder for shape model.")
    parser.add_argument("--tex_model", type=str, default="tencent/Hunyuan3D-2", help="Path to texture generation model.")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps for shape generation.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use.")
    return parser.parse_args()

def find_image(folder, view_names):
    """
    Tries to find an image matching one of the view_names in the folder.
    Returns the path if found, else None.
    Case insensitive search.
    """
    for f in os.listdir(folder):
        name_part = os.path.splitext(f)[0].lower()
        if name_part in view_names and f.lower().endswith(('.png', '.jpg', '.jpeg')):
            return os.path.join(folder, f)
    return None

def main():
    args = parse_args()
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

    # Initialize Models
    print(f"Loading Shape Model: {args.shape_model}...")
    shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.shape_model,
        subfolder=args.shape_subfolder,
        device=args.device
    )
    
    print(f"Loading Texture Model: {args.tex_model}...")
    paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
        args.tex_model,
        device=args.device
    )
    
    rembg = BackgroundRemover()

    # Define view mapping
    # Maps internal key -> list of possible filenames (lowercase)
    view_mapping = {
        "front": ["front", "f"],
        "back": ["back", "b"],
        "left": ["left", "l"],
        "right": ["right", "r"]
    }

    # Walk through input directory
    for root, dirs, files in os.walk(input_root):
        current_path = Path(root)
        
        # Check if this folder contains at least a Front view
        front_img_path = find_image(root, view_mapping["front"])
        
        if not front_img_path:
            continue # Skip if no front view found
            
        print(f"\nProcessing folder: {current_path}")
        
        # Prepare inputs
        images_dict = {}
        processed_images = {}
        
        # Try to find all views
        for key, aliases in view_mapping.items():
            img_path = find_image(root, aliases)
            if img_path:
                print(f"  Found {key}: {img_path}")
                img = Image.open(img_path).convert("RGBA")
                
                # Remove background
                # We do this for all inputs to ensure clean generation
                img_nobg = rembg(img)
                images_dict[key] = img_nobg
                processed_images[key] = img_nobg
        
        if "front" not in images_dict:
            print("  Skipping: Front view processing failed.")
            continue

        # Determine output path
        # Structure: output_root / relative_path / file.glb
        rel_path = current_path.relative_to(input_root)
        target_dir = output_root / rel_path
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Save processed (no bg) images
        no_bg_dir = target_dir / "no_background"
        no_bg_dir.mkdir(exist_ok=True)
        for key, img in processed_images.items():
            save_path = no_bg_dir / f"{key}.png"
            img.save(save_path)
            print(f"  Saved processed image: {save_path}")

        # 1. Generate Shape
        print("  Generating Shape...")
        start_time = time.time()
        mesh = shape_pipeline(
            image=images_dict,
            num_inference_steps=args.steps,
            generator=torch.manual_seed(args.seed),
            output_type='trimesh'
        )[0]
        print(f"  Shape generated in {time.time() - start_time:.2f}s")

        # 2. Generate Texture
        # Note: Paint pipeline primarily uses the front view for conditioning currently
        print("  Generating Texture...")
        start_time = time.time()
        mesh = paint_pipeline(mesh, image=images_dict["front"])
        print(f"  Texture generated in {time.time() - start_time:.2f}s")

        # 3. Export
        output_filename = f"{current_path.name}.glb"
        output_path = target_dir / output_filename
        mesh.export(str(output_path))
        print(f"  Saved model to: {output_path}")

if __name__ == "__main__":
    main()
