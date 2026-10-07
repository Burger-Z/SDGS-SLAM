"""Flow Matching denoiser wrapper for SLAM latent and RGB refinement.

This module provides a thin wrapper around FlowMatchScheduler that integrates
with the SLAM pipeline for optional RGB/latent frame refinement.

For the first integration pass, this is intentionally minimal:
- No neural network model required yet
- Can be enabled/disabled via config
- Compatible with future model-based denoising
"""

import torch
from typing import Optional
from utils.flow_match_scheduler import FlowMatchScheduler


class FlowMatchRefiner:
    """
    Flow Matching refinement for SLAM frames.
    
    Refines RGB or latent tensors using the Flow Matching scheduler.
    Currently supports identity-style (no model) or placeholder updates.
    Real model integration happens in next phase.
    """
    
    def __init__(
        self,
        num_inference_steps: int = 12,
        shift: float = 1.0,
        sigma_max: float = 14.6,
        sigma_min: float = 0.003,
        denoising_strength: float = 1.0,
        extra_one_step: bool = False,
        device: str = "cuda",
    ):
        """
        Initialize Flow Matching refiner.
        
        Args:
            num_inference_steps: Number of denoising steps
            shift: Quality-speed tradeoff (1.0=default, >1.0=quality)
            sigma_max: Maximum noise level
            sigma_min: Minimum noise level
            denoising_strength: Strength of refinement (0-1)
            extra_one_step: Add refinement step at end
            device: Device to run on
        """
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.denoising_strength = denoising_strength
        
        # Initialize scheduler
        self.scheduler = FlowMatchScheduler(
            num_inference_steps=num_inference_steps,
            shift=shift,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            extra_one_step=extra_one_step,
        )
        
        # Configure timesteps
        self.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=denoising_strength,
            training=False,
        )
    
    def refine_rgb(
        self,
        rgb_frame: torch.Tensor,
        model: Optional[torch.nn.Module] = None,
        use_placeholder: bool = True,
    ) -> torch.Tensor:
        """
        Refine an RGB frame using Flow Matching.
        
        Args:
            rgb_frame: Input RGB frame (C, H, W) [0, 1]
            model: Optional denoising model. If None, uses placeholder update.
            use_placeholder: If True and model is None, use identity-style update
            
        Returns:
            Refined RGB frame (C, H, W) [0, 1]
        """
        current = rgb_frame.clone().float()
        
        for timestep in self.scheduler.timesteps:
            if model is not None:
                # Real model path (future work)
                with torch.no_grad():
                    pred = model(current, timestep=timestep)
            else:
                # Placeholder: zero gradient (identity-like)
                # This validates the scheduler wiring without a model
                pred = torch.zeros_like(current)
            
            # Scheduler step
            current = self.scheduler.step(pred, timestep, current)
        
        return current
    
    def refine_latents(
        self,
        latents: torch.Tensor,
        model: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:
        """
        Refine latent features using Flow Matching.
        
        Args:
            latents: Input latent tensor (B, C, H, W) or (C, H, W)
            model: Optional denoising model
            
        Returns:
            Refined latent tensor, same shape as input
        """
        current = latents.clone().float()
        
        for timestep in self.scheduler.timesteps:
            if model is not None:
                with torch.no_grad():
                    pred = model(current, timestep=timestep)
            else:
                pred = torch.zeros_like(current)
            
            current = self.scheduler.step(pred, timestep, current)
        
        return current


def create_flow_refiner(config: dict, device: str = "cuda") -> Optional[FlowMatchRefiner]:
    """
    Factory function to create a Flow Matching refiner from config.
    
    Args:
        config: Configuration dict with 'flow_match' key
        device: Device to run on
        
    Returns:
        FlowMatchRefiner or None if disabled
    """
    if not config.get("use_flow_match", False):
        return None
    
    flow_cfg = config.get("flow_match", {})
    
    refiner = FlowMatchRefiner(
        num_inference_steps=flow_cfg.get("num_inference_steps", 12),
        shift=flow_cfg.get("shift", 1.0),
        sigma_max=flow_cfg.get("sigma_max", 14.6),
        sigma_min=flow_cfg.get("sigma_min", 0.003),
        denoising_strength=flow_cfg.get("denoising_strength", 1.0),
        extra_one_step=flow_cfg.get("extra_one_step", False),
        device=device,
    )
    
    return refiner
