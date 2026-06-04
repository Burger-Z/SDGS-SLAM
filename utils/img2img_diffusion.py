import torch
import torch.nn.functional as F
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
)
from PIL import Image
import numpy as np
from typing import Optional, List
import warnings


class ImageToImageDiffusion:
    """
    Image-to-Image diffusion model for frame enhancement before Gaussian splatting.
    Supports:
    - Standard image-to-image enhancement
    - Depth-guided enhancement using ControlNet
    - Progressive multi-step refinement
    """
    
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-2",
        enable_controlnet: bool = False,
        controlnet_id: str = "lllyasviel/sd-controlnet-depth",
        num_inference_steps: int = 20,
        device: str = "cuda",
        use_fp16: bool = True,
        enable_attention_slicing: bool = True,
        enable_memory_efficient_attention: bool = True,
    ):
        """
        Initialize Image-to-Image diffusion model.
        
        Args:
            model_id: HuggingFace model identifier
            enable_controlnet: Use ControlNet for depth/semantic guidance
            controlnet_id: ControlNet model identifier
            num_inference_steps: Number of diffusion steps (15-30 recommended)
            device: Device to run on
            use_fp16: Use half precision
            enable_attention_slicing: Enable attention slicing for memory efficiency
            enable_memory_efficient_attention: Enable xformers optimization
        """
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.enable_controlnet = enable_controlnet
        
        print(f"Loading Image-to-Image diffusion model: {model_id}")
        
        dtype = torch.float16 if use_fp16 else torch.float32
        
        if enable_controlnet:
            print(f"Loading ControlNet: {controlnet_id}")
            self.controlnet = ControlNetModel.from_pretrained(
                controlnet_id,
                torch_dtype=dtype,
            ).to(device)
            
            self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                model_id,
                controlnet=self.controlnet,
                torch_dtype=dtype,
                safety_checker=None,
            ).to(device)
        else:
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                safety_checker=None,
            ).to(device)
        
        # Memory optimizations
        if enable_attention_slicing:
            self.pipe.enable_attention_slicing()
        
        if enable_memory_efficient_attention:
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except ImportError:
                warnings.warn("xformers not installed, skipping memory efficient attention")
        
        print(f"Image-to-Image diffusion model loaded on {device}")
    
    def enhance_with_img2img(
        self,
        image: torch.Tensor,
        prompt: str = "high quality, detailed scene, sharp focus",
        negative_prompt: str = "blurry, low quality, distorted, artifacts",
        strength: float = 0.6,
        guidance_scale: float = 7.5,
        seed: Optional[int] = 0,
    ) -> torch.Tensor:
        """
        Basic image-to-image enhancement without guidance.
        
        Args:
            image: Input image tensor (C, H, W) [0, 1]
            prompt: Text prompt for positive guidance
            negative_prompt: Text prompt for negative guidance
            strength: Denoising strength (0-1)
                     - Lower (0.3-0.5): subtle enhancement, closer to original
                     - Higher (0.6-0.8): stronger enhancement, more creative
            guidance_scale: Strength of text guidance (7.5-15.0 typical)
            seed: Random seed for reproducibility
            
        Returns:
            Enhanced image tensor (C, H, W) [0, 1]
        """
        image_pil = self._tensor_to_pil(image)
        
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image_pil,
                strength=strength,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None,
            )
            enhanced_pil = result.images[0]
        
        return self._pil_to_tensor(enhanced_pil)
    
    def enhance_with_depth_guidance(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        prompt: str = "high quality, detailed scene, sharp focus",
        negative_prompt: str = "blurry, low quality",
        strength: float = 0.6,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        seed: Optional[int] = 0,
    ) -> torch.Tensor:
        """
        Image-to-image enhancement guided by depth map using ControlNet.
        Preserves depth structure while enhancing appearance.
        
        Args:
            image: Input image tensor (C, H, W) [0, 1]
            depth: Depth map tensor (1, H, W) [0, 1] or normalized
            prompt: Positive prompt
            negative_prompt: Negative prompt
            strength: Denoising strength (0.4-0.7 recommended for depth guidance)
            guidance_scale: Text guidance strength
            controlnet_conditioning_scale: How strongly to apply depth guidance (0-1)
                                          - 0.5-0.7: moderate depth preservation
                                          - 0.8-1.0: strong depth preservation
            seed: Random seed
            
        Returns:
            Enhanced image tensor (C, H, W) [0, 1]
        """
        if not self.enable_controlnet:
            print("Warning: ControlNet not enabled. Using standard img2img enhancement.")
            return self.enhance_with_img2img(image, prompt, negative_prompt, strength, guidance_scale, seed)
        
        image_pil = self._tensor_to_pil(image)
        
        # Process depth for ControlNet (needs to be normalized 0-255)
        depth_np = depth.squeeze(0).cpu().numpy()
        depth_min = depth_np[depth_np > 0].min() if (depth_np > 0).any() else 0
        depth_max = depth_np.max()
        
        if depth_max > depth_min:
            depth_normalized = ((depth_np - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        else:
            depth_normalized = (depth_np * 255).astype(np.uint8)
        
        depth_pil = Image.fromarray(depth_normalized, mode='L').convert('RGB')
        
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image_pil,
                control_image=depth_pil,
                strength=strength,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None,
            )
            enhanced_pil = result.images[0]
        
        return self._pil_to_tensor(enhanced_pil)
    
    def enhance_progressive(
        self,
        image: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
        num_steps: int = 3,
        strength_schedule: Optional[List[float]] = None,
        prompt: str = "high quality, detailed scene",
    ) -> torch.Tensor:
        """
        Progressive image-to-image enhancement with decreasing strength.
        Refines quality iteratively while staying close to original.
        
        Args:
            image: Input image tensor (C, H, W) [0, 1]
            depth: Optional depth map for depth-guided enhancement
            num_steps: Number of progressive enhancement steps
            strength_schedule: List of strengths for each step
                             If None, uses default decreasing schedule
            prompt: Enhancement prompt
            
        Returns:
            Progressively enhanced image (C, H, W) [0, 1]
        """
        if strength_schedule is None:
            # Default: decreasing strength schedule
            strength_schedule = [0.4 + (0.3 * (1 - i / num_steps)) for i in range(num_steps)]
        
        current_image = image.clone()
        
        for step, strength in enumerate(strength_schedule):
            print(f"Progressive enhancement step {step + 1}/{num_steps} (strength={strength:.2f})")
            
            if depth is not None and self.enable_controlnet:
                current_image = self.enhance_with_depth_guidance(
                    current_image,
                    depth,
                    prompt=prompt,
                    strength=strength,
                    seed=step,
                )
            else:
                current_image = self.enhance_with_img2img(
                    current_image,
                    prompt=prompt,
                    strength=strength,
                    seed=step,
                )
        
        return current_image
    
    def blend_enhancement(
        self,
        original: torch.Tensor,
        enhanced: torch.Tensor,
        blend_strength: float = 0.7,
    ) -> torch.Tensor:
        """
        Blend original and enhanced images for fine control.
        
        Args:
            original: Original image (C, H, W) [0, 1]
            enhanced: Enhanced image (C, H, W) [0, 1]
            blend_strength: How much to blend (0-1)
                           0.0: return original
                           1.0: return enhanced
                           0.5: 50% blend
            
        Returns:
            Blended image (C, H, W) [0, 1]
        """
        return original * (1 - blend_strength) + enhanced * blend_strength
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert torch tensor (C, H, W) [0, 1] to PIL Image."""
        array = tensor.cpu().numpy()
        
        if array.shape[0] == 1:
            array = np.repeat(array, 3, axis=0)
        if array.shape[0] > 3:
            array = array[:3]
        
        array = (np.clip(array, 0, 1) * 255).astype(np.uint8)
        if array.shape[0] == 3:
            array = np.transpose(array, (1, 2, 0))
        
        return Image.fromarray(array, mode='RGB')
    
    def _pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to torch tensor (C, H, W) [0, 1]."""
        array = np.array(pil_image).astype(np.float32) / 255.0
        
        if len(array.shape) == 3:
            if array.shape[2] == 4:
                array = array[:, :, :3]
            tensor = torch.from_numpy(np.transpose(array, (2, 0, 1)))
        else:
            tensor = torch.from_numpy(array).unsqueeze(0)
        
        return tensor.to(self.device)


def create_img2img_diffusion(config: dict) -> Optional[ImageToImageDiffusion]:
    """
    Factory function to create Image-to-Image diffusion from config.
    
    Args:
        config: Configuration dictionary with keys:
                - 'use_diffusion': bool
                - 'diffusion': dict with model config
                
    Returns:
        ImageToImageDiffusion instance or None
    """
    if not config.get('use_diffusion', False):
        return None
    
    diffusion_config = config.get('diffusion', {})
    img2img_config = diffusion_config.get('img2img', {})
    
    return ImageToImageDiffusion(
        model_id=img2img_config.get('model_id', 'stabilityai/stable-diffusion-2'),
        enable_controlnet=img2img_config.get('enable_controlnet', False),
        controlnet_id=img2img_config.get('controlnet_id', 'lllyasviel/sd-controlnet-depth'),
        num_inference_steps=img2img_config.get('num_inference_steps', 20),
        device=img2img_config.get('device', 'cuda'),
        use_fp16=img2img_config.get('use_fp16', True),
        enable_attention_slicing=img2img_config.get('enable_attention_slicing', True),
        enable_memory_efficient_attention=img2img_config.get('enable_memory_efficient_attention', True),
    )
