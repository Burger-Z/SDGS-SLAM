# Configuration Example for SDGS-SLAM with Image-to-Image Diffusion
# This shows how to add diffusion preprocessing to your SLAM config

# ============= DIFFUSION MODEL CONFIGURATION =============

use_diffusion = True  # Enable/disable diffusion preprocessing

diffusion_config = dict(
    # Base model selection
    model_id="stabilityai/stable-diffusion-2",
    # Alternatives:
    # - "runwayml/stable-diffusion-v1-5" (faster, ~4GB VRAM)
    # - "stabilityai/stable-diffusion-2-1" (higher quality)
    # - "SG161222/Realistic_Vision_V6.0_B1_noVAE" (photorealistic)
    
    # Image-to-Image specific settings
    img2img=dict(
        # Enhancement mode: 'img2img', 'depth_guided', 'progressive'
        mode='depth_guided',  # RECOMMENDED for SLAM
        
        # ControlNet settings (for depth_guided mode)
        enable_controlnet=True,
        controlnet_id='lllyasviel/sd-controlnet-depth',
        # Other ControlNet options:
        # - "lllyasviel/sd-controlnet-normal" (surface normals)
        # - "lllyasviel/sd-controlnet-canny" (edge detection)
        # - "lllyasviel/sd-controlnet-mlsd" (straight lines)
        
        # Inference settings
        num_inference_steps=20,  # 15-30 (lower=faster, higher=quality)
        # Recommended: 20 for balance, 15 for speed, 30 for quality
        
        device='cuda',  # 'cuda' or 'cpu'
        use_fp16=True,  # Half precision (saves ~50% VRAM)
        enable_attention_slicing=True,  # Memory efficiency
        enable_memory_efficient_attention=True,  # xformers optimization
    ),
    
    # When to apply diffusion
    apply_before_tracking=True,   # Apply during tracking phase
    apply_before_mapping=False,   # Apply during mapping phase (slower)
    
    # Prompts for guidance (customize for your scene)
    prompt='high quality, detailed, sharp focus, clear, photorealistic indoor scene',
    negative_prompt='blurry, low quality, distorted, artifacts, noise, deformed, shadows',
    
    # Diffusion strength (0-1)
    strength=0.6,
    # - 0.3-0.4: Subtle enhancement, very close to original
    # - 0.5-0.6: Moderate enhancement (RECOMMENDED)
    # - 0.7-0.8: Strong enhancement, more creative
    # - 0.9+: Very strong, may distort geometry
    
    # Text guidance scale (typically 7.0-15.0)
    guidance_scale=7.5,
    # - 5.0-7.5: Conservative guidance
    # - 7.5-10.0: Moderate guidance (RECOMMENDED: 7.5)
    # - 10.0-15.0: Strong guidance (may over-constrain)
    
    # For depth_guided mode: How strongly to apply depth constraint
    controlnet_scale=0.7,
    # - 0.5-0.7: Moderate depth preservation
    # - 0.7-0.9: Strong depth preservation (RECOMMENDED: 0.7)
    # - 0.9-1.0: Very strong, minimal appearance change
    
    # For progressive mode: Multi-step refinement
    num_progressive_steps=2,
    # - 2: Fast (2x forward passes)
    # - 3: Better quality (3x forward passes)
)

# =========================================================

# Usage: Add to your main config dict
"""
config = dict(
    # ... your existing config ...
    
    use_diffusion=use_diffusion,
    diffusion=diffusion_config,
    
    # ... rest of config ...
)
"""
