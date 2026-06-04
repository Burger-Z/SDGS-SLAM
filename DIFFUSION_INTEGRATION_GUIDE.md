# Image-to-Image Diffusion Integration Guide for SDGS-SLAM

## Overview

This guide explains how to integrate the Image-to-Image diffusion model for frame enhancement **before Gaussian splatting rendering begins**.

The diffusion model enhances RGB frames using Stable Diffusion with optional depth-guided ControlNet, improving reconstruction quality while preserving spatial structure.

## Features

- ✅ Standard image-to-image enhancement
- ✅ Depth-guided enhancement using ControlNet (preserves geometry)
- ✅ Progressive multi-step refinement
- ✅ Memory-efficient (FP16, attention slicing, xformers)
- ✅ Flexible prompting for scene-specific enhancement

## Installation

```bash
pip install diffusers transformers safetensors
pip install xformers  # Optional but recommended for speed
```

## Configuration

### Basic Setup (in your config file)

Add to your config dict (e.g., `configs/replica/slam.py`):

```python
config = dict(
    # ... existing config ...
    
    use_diffusion=True,
    diffusion=dict(
        model_id="stabilityai/stable-diffusion-2",
        img2img=dict(
            mode='depth_guided',  # 'img2img', 'depth_guided', 'progressive'
            enable_controlnet=True,
            controlnet_id='lllyasviel/sd-controlnet-depth',
            num_inference_steps=20,  # 15-30 (lower=faster)
            device='cuda',
            use_fp16=True,
        ),
        apply_before_tracking=True,
        prompt='high quality, detailed, sharp focus, photorealistic',
        negative_prompt='blurry, low quality, artifacts',
        strength=0.6,  # 0.4-0.7 recommended
        guidance_scale=7.5,
        controlnet_scale=0.7,  # Depth preservation strength
    ),
    
    # ... rest of config ...
)
```

## Integration Steps

### Step 1: Add Import to `scripts/slam.py`

At the top of the file with other imports:

```python
from utils.img2img_diffusion import ImageToImageDiffusion, create_img2img_diffusion
```

### Step 2: Initialize Diffusion Model

In the `rgbd_slam()` function, after loading config (around line 582):

```python
# ===== DIFFUSION MODEL INITIALIZATION =====
use_diffusion = config.get('use_diffusion', False)
if use_diffusion:
    print("Initializing Image-to-Image Diffusion Model...")
    diffusion_model = create_img2img_diffusion(config)
else:
    diffusion_model = None
# ==========================================
```

### Step 3: Apply Diffusion Before Gaussian Splatting

In the main loop (around line 735-750), after loading frame:

```python
# Load RGBD frames
if load_semantics:
    color, depth, _, gt_pose, semantic_id, semantic_color = dataset[time_idx]
else:
    color, depth, _, gt_pose = dataset[time_idx]

# Process poses
gt_w2c = torch.linalg.inv(gt_pose)

# Process RGB-D Data
color = color.permute(2, 0, 1) / 255  # (H, W, C) -> (C, H, W)
depth = depth.permute(2, 0, 1)         # (H, W, 1) -> (1, H, W)

# ===== APPLY IMAGE-TO-IMAGE DIFFUSION =====
if diffusion_model is not None and config['diffusion'].get('apply_before_tracking', True):
    print(f"[Frame {time_idx}] Applying diffusion enhancement...")
    mode = config['diffusion'].get('img2img', {}).get('mode', 'img2img')
    
    try:
        if mode == 'depth_guided':
            color = diffusion_model.enhance_with_depth_guidance(
                color, depth,
                prompt=config['diffusion'].get('prompt', 'high quality scene'),
                negative_prompt=config['diffusion'].get('negative_prompt', ''),
                strength=config['diffusion'].get('strength', 0.6),
                guidance_scale=config['diffusion'].get('guidance_scale', 7.5),
                controlnet_conditioning_scale=config['diffusion'].get('controlnet_scale', 0.7),
                seed=time_idx,
            )
        elif mode == 'progressive':
            color = diffusion_model.enhance_progressive(
                color, depth=depth,
                num_steps=config['diffusion'].get('img2img', {}).get('num_progressive_steps', 2),
                prompt=config['diffusion'].get('prompt', 'high quality scene'),
            )
        else:  # 'img2img'
            color = diffusion_model.enhance_with_img2img(
                color,
                prompt=config['diffusion'].get('prompt', 'high quality scene'),
                negative_prompt=config['diffusion'].get('negative_prompt', ''),
                strength=config['diffusion'].get('strength', 0.6),
                guidance_scale=config['diffusion'].get('guidance_scale', 7.5),
                seed=time_idx,
            )
        print(f"✓ Enhancement completed")
    except Exception as e:
        print(f"✗ Error during diffusion: {e}")
        print(f"Continuing with original frame...")
# ==========================================
```

## Enhancement Modes

### 1. Standard Image-to-Image (img2img)
```python
mode='img2img'
strength=0.5  # Subtle enhancement
```
- Basic enhancement without guidance
- Fastest option
- Best for general quality improvement

### 2. Depth-Guided (RECOMMENDED for SLAM)
```python
mode='depth_guided'
enable_controlnet=True
strength=0.6
controlnet_scale=0.7  # How strongly to preserve depth
```
- Uses ControlNet to preserve depth structure
- Enhances appearance while maintaining geometry
- Critical for maintaining point cloud accuracy

### 3. Progressive Enhancement
```python
mode='progressive'
num_progressive_steps=2
strength_schedule=[0.7, 0.5]  # Decreasing strength
```
- Multi-step refinement
- More stable results
- Better quality but slower

## Hyperparameter Tuning

### Strength (0-1)
- **0.3-0.4**: Subtle enhancement, very close to original
- **0.5-0.6**: Moderate enhancement (RECOMMENDED)
- **0.7-0.8**: Strong enhancement, more creative
- **0.9+**: Very strong, may distort geometry

### Guidance Scale
- **5.0-7.5**: Conservative, follows prompt loosely
- **7.5-10.0**: Moderate guidance (RECOMMENDED: 7.5)
- **10.0-15.0**: Strong guidance, may over-constrain

### ControlNet Conditioning Scale (depth guidance)
- **0.5-0.7**: Moderate depth preservation
- **0.7-0.9**: Strong depth preservation (RECOMMENDED: 0.7)
- **0.9-1.0**: Very strong, minimal appearance change

## Model Options

### Base Models
- `"stabilityai/stable-diffusion-2"` - Default, good balance
- `"runwayml/stable-diffusion-v1-5"` - Faster, lower VRAM
- `"stabilityai/stable-diffusion-2-1"` - Higher quality

### ControlNet Options
- `"lllyasviel/sd-controlnet-depth"` - Depth preservation (RECOMMENDED)
- `"lllyasviel/sd-controlnet-normal"` - Surface normals
- `"lllyasviel/sd-controlnet-canny"` - Edge detection

## Memory Management

### Enable for Low-VRAM Systems
```python
use_fp16=True,  # Half precision
enable_attention_slicing=True,  # Memory efficient attention
enable_memory_efficient_attention=True,  # xformers optimization
```

### Reduce Inference Steps
```python
num_inference_steps=15  # Faster but lower quality (default: 20)
```

### Reduce Model Size
```python
model_id="runwayml/stable-diffusion-v1-5"  # Smaller than SD2
```

## Performance Tips

1. **Use depth-guided mode** for better geometry preservation
2. **Lower inference steps** (15-20) for speed, increase to 30 for quality
3. **Use seed=time_idx** for temporal consistency across frames
4. **Enable xformers** for 20-30% speedup
5. **Use FP16** to reduce VRAM by ~50%

## Troubleshooting

### Out of Memory (OOM)
- Reduce `num_inference_steps` (try 15)
- Enable `use_fp16=True`
- Enable `enable_attention_slicing=True`
- Use smaller model: `"runwayml/stable-diffusion-v1-5"`

### Slow Inference
- Install `xformers`: `pip install xformers`
- Reduce `num_inference_steps`
- Use `use_fp16=True`

### Artifacts in Output
- Reduce `strength` (try 0.5)
- Reduce `guidance_scale` (try 7.5)
- Disable depth guidance for problematic frames

### Geometry Distortion
- Increase `controlnet_conditioning_scale` (try 0.9)
- Reduce `strength` (try 0.5)
- Use `progressive` mode instead

## Example Configurations

### High Quality (Slow)
```python
use_diffusion=True,
diffusion=dict(
    model_id="stabilityai/stable-diffusion-2-1",
    img2img=dict(
        mode='depth_guided',
        enable_controlnet=True,
        num_inference_steps=30,
        use_fp16=True,
    ),
    strength=0.7,
    guidance_scale=10.0,
    controlnet_scale=0.8,
)
```

### Fast (Low Quality)
```python
use_diffusion=True,
diffusion=dict(
    model_id="runwayml/stable-diffusion-v1-5",
    img2img=dict(
        mode='img2img',
        enable_controlnet=False,
        num_inference_steps=15,
        use_fp16=True,
    ),
    strength=0.5,
    guidance_scale=7.5,
)
```

### Balanced (Recommended)
```python
use_diffusion=True,
diffusion=dict(
    model_id="stabilityai/stable-diffusion-2",
    img2img=dict(
        mode='depth_guided',
        enable_controlnet=True,
        num_inference_steps=20,
        use_fp16=True,
    ),
    strength=0.6,
    guidance_scale=7.5,
    controlnet_scale=0.7,
)
```

## Usage

```bash
# Run SLAM with diffusion enhancement
python scripts/slam.py configs/replica/slam.py
```

The diffusion model will automatically enhance each frame **before** the Gaussian splatting rendering pipeline begins.

## References

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [ControlNet](https://github.com/lllyasviel/ControlNet)
- [Diffusers Library](https://huggingface.co/docs/diffusers)
