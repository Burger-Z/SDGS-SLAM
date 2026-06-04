# Step-by-Step Integration of Image-to-Image Diffusion into SDGS-SLAM

## Overview

This guide provides exact code changes needed to integrate the diffusion model into your SLAM pipeline.

## Step 1: Add Import to `scripts/slam.py`

### Location
Near the top of `scripts/slam.py`, with other utility imports (around line 38-42).

### Add This Line
```python
from utils.img2img_diffusion import ImageToImageDiffusion, create_img2img_diffusion
```

### Complete Import Section (Example)
```python
from utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
    transformed_semantics2rendervar, transform_to_frame, l1_loss_v1, matrix_to_quaternion
)
from utils.img2img_diffusion import ImageToImageDiffusion, create_img2img_diffusion  # ADD THIS LINE
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians, densify
```

---

## Step 2: Initialize Diffusion Model in `rgbd_slam()` Function

### Location
Inside the `rgbd_slam(config: dict)` function, after the config is loaded and printed (around line 580-590).

### Add This Code Block
```python
def rgbd_slam(config: dict):
    # Print Config
    print("Loaded Config:")
    # ... existing config printout code ...
    
    # ===== DIFFUSION MODEL INITIALIZATION =====
    use_diffusion = config.get('use_diffusion', False)
    if use_diffusion:
        print("Initializing Image-to-Image Diffusion Model for frame preprocessing...")
        diffusion_model = create_img2img_diffusion(config)
        if diffusion_model is None:
            use_diffusion = False
            print("Warning: Diffusion model failed to load")
    else:
        diffusion_model = None
    # =========================================
    
    # Create Output Directories
    # ... rest of function ...
```

---

## Step 3: Apply Diffusion in Main Loop

### Location
Inside the main frame processing loop, after loading and processing the RGB-D frames (around line 735-750).

### Find This Code
```python
for time_idx in tqdm(range(checkpoint_time_idx, num_frames)):
    # Load RGBD frames incrementally instead of all frames
    if load_semantics:
        color, depth, _, gt_pose, semantic_id, semantic_color = dataset[time_idx]
    else:
        color, depth, _, gt_pose = dataset[time_idx]
    
    # Process poses
    gt_w2c = torch.linalg.inv(gt_pose)
    
    # Process RGB-D Data
    color = color.permute(2, 0, 1) / 255
    depth = depth.permute(2, 0, 1)
    gt_w2c_all_frames.append(gt_w2c)
```

### Replace With This
```python
for time_idx in tqdm(range(checkpoint_time_idx, num_frames)):
    # Load RGBD frames incrementally instead of all frames
    if load_semantics:
        color, depth, _, gt_pose, semantic_id, semantic_color = dataset[time_idx]
    else:
        color, depth, _, gt_pose = dataset[time_idx]
    
    # Process poses
    gt_w2c = torch.linalg.inv(gt_pose)
    
    # Process RGB-D Data
    color = color.permute(2, 0, 1) / 255
    depth = depth.permute(2, 0, 1)
    
    # ===== APPLY IMAGE-TO-IMAGE DIFFUSION PREPROCESSING =====
    if diffusion_model is not None and use_diffusion and config['diffusion'].get('apply_before_tracking', True):
        try:
            print(f"[Frame {time_idx}] Applying Image-to-Image diffusion enhancement...")
            mode = config['diffusion'].get('img2img', {}).get('mode', 'img2img')
            
            if mode == 'depth_guided':
                # Depth-guided enhancement using ControlNet
                color = diffusion_model.enhance_with_depth_guidance(
                    color,
                    depth,
                    prompt=config['diffusion'].get('prompt', 'high quality scene'),
                    negative_prompt=config['diffusion'].get('negative_prompt', ''),
                    strength=config['diffusion'].get('strength', 0.6),
                    guidance_scale=config['diffusion'].get('guidance_scale', 7.5),
                    controlnet_conditioning_scale=config['diffusion'].get('controlnet_scale', 0.7),
                    seed=time_idx,
                )
            elif mode == 'progressive':
                # Progressive multi-step enhancement
                color = diffusion_model.enhance_progressive(
                    color,
                    depth=depth,
                    num_steps=config['diffusion'].get('img2img', {}).get('num_progressive_steps', 2),
                    prompt=config['diffusion'].get('prompt', 'high quality scene'),
                )
            else:
                # Standard image-to-image enhancement
                color = diffusion_model.enhance_with_img2img(
                    color,
                    prompt=config['diffusion'].get('prompt', 'high quality scene'),
                    negative_prompt=config['diffusion'].get('negative_prompt', ''),
                    strength=config['diffusion'].get('strength', 0.6),
                    guidance_scale=config['diffusion'].get('guidance_scale', 7.5),
                    seed=time_idx,
                )
            print(f"✓ Diffusion enhancement completed for frame {time_idx}")
        except Exception as e:
            print(f"✗ Error during diffusion preprocessing: {e}")
            print(f"Continuing with original frame...")
    # =====================================================
    
    gt_w2c_all_frames.append(gt_w2c)
```

---

## Step 4: Update Config File

### File
Your config file (e.g., `configs/replica/slam.py`)

### Add to `config` dict

```python
config = dict(
    # ... existing config ...
    
    # ============= IMAGE-TO-IMAGE DIFFUSION =============
    use_diffusion=True,
    diffusion=dict(
        model_id="stabilityai/stable-diffusion-2",
        img2img=dict(
            mode='depth_guided',  # 'img2img', 'depth_guided', 'progressive'
            enable_controlnet=True,
            controlnet_id='lllyasviel/sd-controlnet-depth',
            num_inference_steps=20,
            device='cuda',
            use_fp16=True,
            enable_attention_slicing=True,
            enable_memory_efficient_attention=True,
        ),
        apply_before_tracking=True,
        prompt='high quality, detailed, sharp focus, clear, photorealistic scene',
        negative_prompt='blurry, low quality, distorted, artifacts, noise',
        strength=0.6,
        guidance_scale=7.5,
        controlnet_scale=0.7,
    ),
    # ===================================================
    
    # ... rest of config ...
)
```

---

## Step 5: Install Dependencies

```bash
pip install diffusers transformers safetensors
pip install xformers  # Optional but highly recommended
```

---

## Step 6: Run with Diffusion

```bash
python scripts/slam.py configs/replica/slam.py
```

---

## Verification

You should see output like:

```
Loading Image-to-Image diffusion model: stabilityai/stable-diffusion-2
Loading ControlNet: lllyasviel/sd-controlnet-depth
Image-to-Image diffusion model loaded on cuda

[Frame 0] Applying Image-to-Image diffusion enhancement...
✓ Diffusion enhancement completed for frame 0

[Frame 1] Applying Image-to-Image diffusion enhancement...
✓ Diffusion enhancement completed for frame 1
```

---

## Troubleshooting

### Issue: "use_diffusion not defined"
**Solution:** Make sure you added the initialization block in Step 2.

### Issue: CUDA Out of Memory
**Solution:** 
- Set `num_inference_steps=15` (instead of 20)
- Set `use_fp16=True`
- Use `model_id="runwayml/stable-diffusion-v1-5"` (smaller)

### Issue: Very Slow Processing
**Solution:**
- Install xformers: `pip install xformers`
- Reduce `num_inference_steps` to 15
- Disable diffusion during mapping: `apply_before_mapping=False`

### Issue: Distorted Reconstruction
**Solution:**
- Reduce `strength` to 0.5
- Increase `controlnet_scale` to 0.9 (preserves depth more)
- Disable diffusion: `use_diffusion=False`

---

## Next Steps

1. Experiment with different prompts for your scene
2. Adjust strength and guidance_scale for optimal results
3. Monitor memory usage and adjust inference steps if needed
4. Compare reconstruction quality with and without diffusion

For more details, see `DIFFUSION_INTEGRATION_GUIDE.md`
