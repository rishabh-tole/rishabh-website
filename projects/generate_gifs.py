import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath("."))

from models import SlotBERT

# --- Configuration ---
CHECKPOINT_PATH = "checkpoints/best_model.pt"
VIDEO_DIR = "data/movi_a/test/videos"
OUTPUT_DIR = "attempt1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_VIDEOS = 3

# Define a color palette for slot visualization (7 slots)
SLOT_COLORS = np.array([
    [255, 0, 0],    # Red
    [0, 255, 0],    # Green
    [0, 0, 255],    # Blue
    [255, 255, 0],  # Yellow
    [255, 0, 255],  # Magenta
    [0, 255, 255],  # Cyan
    [128, 128, 128] # Gray
], dtype=np.uint8)

LABEL_HEIGHT = 20  # Height for text labels


def add_label_to_image(img_array, label):
    """Add a text label above an image."""
    H, W = img_array.shape[:2]
    
    # Create a new image with space for label
    labeled = np.ones((H + LABEL_HEIGHT, W, 3), dtype=np.uint8) * 255  # White background
    labeled[LABEL_HEIGHT:, :, :] = img_array
    
    # Convert to PIL to draw text
    pil_img = Image.fromarray(labeled)
    draw = ImageDraw.Draw(pil_img)
    
    # Use default font (or specify a path to a .ttf file)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    # Get text bounding box for centering
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    x = (W - text_width) // 2
    
    draw.text((x, 2), label, fill=(0, 0, 0), font=font)
    
    return np.array(pil_img)


def create_gif(frames, path, fps=4):
    """Save a list of frames as a GIF using PIL."""
    print(f"Saving GIF to {path}...")
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(path, save_all=True, append_images=pil_frames[1:], duration=int(1000/fps), loop=0)


def create_combined_visualization(frame, masks, recon):
    """
    Create a grid visualization for one frame.
    Layout (5 columns x 2 rows):
        Row 1: Original | Segmentation | Reconstruction | Slot 1 | Slot 2
        Row 2: Slot 3   | Slot 4       | Slot 5         | Slot 6 | Slot 7
    """
    H, W = frame.shape[:2]
    K = masks.shape[0]
    
    panels = []
    labels = []
    
    # 1. Original
    panels.append(frame)
    labels.append("Original")
    
    # 2. Argmax Segmentation (colored)
    seg_map = masks.argmax(axis=0)
    seg_colored = SLOT_COLORS[seg_map]
    panels.append(seg_colored)
    labels.append("Segmentation")
    
    # 3. Reconstruction
    recon_uint8 = (np.clip(recon, 0, 1) * 255).astype(np.uint8)
    panels.append(recon_uint8)
    labels.append("Recon")
    
    # 4-10. Each Slot (masked original)
    for k in range(K):
        m = masks[k, :, :]
        masked_frame = (frame.astype(np.float32) * m[:, :, None]).astype(np.uint8)
        panels.append(masked_frame)
        labels.append(f"Slot {k+1}")
    
    # Add labels to each panel
    labeled_panels = [add_label_to_image(p, l) for p, l in zip(panels, labels)]
    
    # Arrange in a 2x5 grid
    # Row 1: indices 0, 1, 2, 3, 4
    # Row 2: indices 5, 6, 7, 8, 9
    row1 = np.concatenate(labeled_panels[0:5], axis=1)
    row2 = np.concatenate(labeled_panels[5:10], axis=1)
    
    combined = np.concatenate([row1, row2], axis=0)
    return combined


def main():
    print(f"🔧 Using device: {DEVICE}")
    
    # 1. Load Model
    print("📥 Loading model...")
    model = SlotBERT(
        num_slots=7,
        slot_dim=64,
        encoder_dim=384,
        use_mobile_encoder=False,
        use_simple_encoder=True,
        max_frames=32
    ).to(DEVICE)
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Checkpoint not found at {CHECKPOINT_PATH}")
        return

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    state_dict = checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']

    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v
            
    try:
        model.load_state_dict(clean_state_dict)
        print("✅ Weights loaded successfully.")
    except Exception as e:
        print(f"⚠️ Strict loading failed: {e}. Trying strict=False...")
        model.load_state_dict(clean_state_dict, strict=False)

    model.eval()

    # 2. Process Videos
    video_dirs = sorted([d for d in Path(VIDEO_DIR).iterdir() if d.is_dir()])
    
    print(f"📂 Found {len(video_dirs)} video directories in {VIDEO_DIR}")
    
    for i, video_path in enumerate(video_dirs[:NUM_VIDEOS]):
        print(f"\n🎬 Processing Video {i+1}: {video_path.name}")
        
        try:
            frame_files = sorted(list(video_path.glob("*.png")) + list(video_path.glob("*.jpg")))
            
            if not frame_files:
                print("No frames found.")
                continue
                
            frames = []
            for frame_file in frame_files:
                img = Image.open(frame_file).convert('RGB').resize((128, 128))
                frames.append(np.array(img))
            
            if len(frames) == 0:
                continue

            # Preprocess
            frames_np = np.array(frames) / 255.0
            frames_tensor = torch.tensor(frames_np).permute(0, 3, 1, 2).float()
            
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            norm_frames = (frames_tensor - mean) / std
            
            input_tensor = norm_frames.unsqueeze(0).to(DEVICE)
            
            # Inference
            with torch.no_grad():
                outputs = model(input_tensor, mask_ratio=0.0)
            
            recon = outputs['reconstruction'][0].permute(0, 2, 3, 1).cpu().numpy()
            masks = outputs['masks'][0].cpu().numpy()
            
            # Create combined visualization GIF
            combined_frames = []
            for t in range(len(frames)):
                combined = create_combined_visualization(frames[t], masks[t], recon[t])
                combined_frames.append(combined)
                
            create_gif(combined_frames, f"{OUTPUT_DIR}/video_{i}_combined.gif")
            
            print("✅ Done.")

        except Exception as e:
            print(f"❌ Failed to process {video_path}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
