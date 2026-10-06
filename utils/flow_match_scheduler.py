"""FlowMatch scheduler for adaptive diffusion timestep scheduling.

Based on Wan2.2 Turbo's FlowMatch scheduler, optimized for few-step inference.
References:
  - FlowMatch: https://arxiv.org/abs/2309.16443
  - Turbo: Stable Diffusion acceleration with flow matching
"""

import torch
import math
from typing import Optional, List


class FlowMatchScheduler:
    """
    Flow Matching scheduler for continuous-time diffusion models.
    
    Replaces DDIMScheduler with continuous sigma scheduling for better quality
    at low step counts (6-20 steps). Particularly effective with Turbo LoRA.
    
    Key features:
    - Continuous time parametrization (better than discrete timesteps)
    - Optimized sigma schedule for few-step inference
    - Configurable shift for different quality/speed tradeoffs
    - Compatible with ControlNet guidance
    """
    
    def __init__(
        self,
        num_inference_steps: int = 20,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,  # 1.0 = standard, >1.0 = favor high quality
        sigma_max: float = 14.6,  # Max noise level (typically around 14.6 for SD)
        sigma_min: float = 0.003,  # Min noise level (near 0 for clean output)
        inverse_timesteps: bool = False,  # Reverse sigma order
        extra_one_step: bool = False,  # Add extra step for refinement
        reverse_sigmas: bool = False,  # Reverse sigma schedule
        exponential_shift: bool = False,  # Use exponential time warping
        exponential_shift_mu: Optional[float] = None,  # Exponential shift parameter
        shift_terminal: Optional[float] = None,  # Terminal shift for trajectory
    ):
        """
        Initialize FlowMatch scheduler.
        
        Args:
            num_inference_steps: Number of denoising steps (6-50 recommended)
            num_train_timesteps: Number of training timesteps (typically 1000)
            shift: Quality-speed tradeoff (1.0=default, 1.5-3.0=higher quality)
            sigma_max: Maximum noise level
            sigma_min: Minimum noise level  
            inverse_timesteps: Reverse the timestep order
            extra_one_step: Add one extra step at the end
            reverse_sigmas: Reverse sigma values
            exponential_shift: Use exponential time warping
            exponential_shift_mu: Exponential warping parameter
            shift_terminal: Terminal shift value
        """
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.exponential_shift = exponential_shift
        self.exponential_shift_mu = exponential_shift_mu
        self.shift_terminal = shift_terminal
        
        # Will be set by set_timesteps()
        self.sigmas = None
        self.timesteps = None
        self.training = False
        self.linear_timesteps_weights = None
        
        # Initialize timesteps
        self.set_timesteps(num_inference_steps)
    
    def set_timesteps(
        self,
        num_inference_steps: int = 20,
        denoising_strength: float = 1.0,
        training: bool = False,
        shift: Optional[float] = None,
        dynamic_shift_len: Optional[int] = None,
    ):
        """
        Compute the timestep schedule for inference or training.
        
        Args:
            num_inference_steps: Number of denoising steps
            denoising_strength: Strength of denoising (0-1, affects starting sigma)
            training: Whether in training mode (affects weighting)
            shift: Optional override for shift parameter
            dynamic_shift_len: For exponential shift, sequence length for dynamic adjustment
        """
        if shift is not None:
            self.shift = shift
        
        # Compute starting sigma based on denoising strength
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        
        # Create sigma schedule
        if self.extra_one_step:
            # Add extra step for refinement
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        
        # Reverse sigmas if requested
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        
        # Apply exponential shift if enabled
        if self.exponential_shift:
            mu = self.calculate_shift(dynamic_shift_len) if dynamic_shift_len is not None else self.exponential_shift_mu
            # Exponential warping: sigma' = e^(mu * log(sigma))
            self.sigmas = torch.exp(mu * torch.log(self.sigmas))
        else:
            # Standard shift: scale sigmas by shift factor
            self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        
        # Apply terminal shift if specified
        if self.shift_terminal is not None:
            one_minus_z = 1 - self.sigmas
            scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
            self.sigmas = 1 - (one_minus_z / scale_factor)
        
        # Reverse sigmas if requested
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        
        # Compute training weights if in training mode
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            # Flow-matching training weights: emphasize mid-range timesteps
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighting = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighting
            self.training = True
        else:
            self.training = False
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: float,
        sample: torch.Tensor,
        to_final: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Perform one denoising step.
        
        Args:
            model_output: Model prediction (predicted noise or velocity)
            timestep: Current timestep
            sample: Current noisy sample
            to_final: Whether this is the final step
            
        Returns:
            Updated sample for next step
        """
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu().item()
        
        # Find closest sigma for this timestep
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        
        # Determine next sigma
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_next = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_next = self.sigmas[timestep_id + 1]
        
        # Flow matching update: x_{t-1} = x_t + (sigma_next - sigma) * model_output
        prev_sample = sample + model_output * (sigma_next - sigma)
        
        return prev_sample
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timestep: float,
    ) -> torch.Tensor:
        """Add noise to samples according to sigma schedule."""
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu().item()
        
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        
        # Add noise: x_noisy = (1 - sigma) * x_original + sigma * noise
        sample = (1 - sigma) * original_samples + sigma * noise
        
        return sample
    
    def training_target(self, sample: torch.Tensor, noise: torch.Tensor, timestep: float) -> torch.Tensor:
        """Compute training target (flow matching predicts noise)."""
        return noise - sample
    
    def denoised_sample(self, prediction: torch.Tensor, noise: torch.Tensor, timestep: float) -> torch.Tensor:
        """Compute denoised sample from model prediction."""
        return noise - prediction
    
    def training_weight(self, timestep: float) -> torch.Tensor:
        """Get training weight for this timestep."""
        if not self.training or self.linear_timesteps_weights is None:
            return torch.tensor(1.0)
        
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        return self.linear_timesteps_weights[timestep_id]
    
    def calculate_shift(
        self,
        image_seq_len: int,
        base_seq_len: int = 256,
        max_seq_len: int = 8192,
        base_shift: float = 0.5,
        max_shift: float = 0.9,
    ) -> float:
        """
        Calculate dynamic shift based on sequence length (for variable-length inputs).
        
        Args:
            image_seq_len: Current sequence length
            base_seq_len: Base sequence length
            max_seq_len: Maximum sequence length
            base_shift: Shift at base length
            max_shift: Shift at maximum length
            
        Returns:
            Dynamic shift factor
        """
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu


# Configuration presets for common use cases
FLOWMATCH_PRESETS = {
    "quality": {
        "num_inference_steps": 20,
        "shift": 1.5,
        "sigma_max": 14.6,
        "sigma_min": 0.003,
    },
    "balanced": {
        "num_inference_steps": 15,
        "shift": 1.0,
        "sigma_max": 14.6,
        "sigma_min": 0.003,
    },
    "turbo_fast": {
        "num_inference_steps": 8,
        "shift": 1.0,
        "sigma_max": 14.6,
        "sigma_min": 0.003,
    },
    "turbo_quality": {
        "num_inference_steps": 12,
        "shift": 1.2,
        "sigma_max": 14.6,
        "sigma_min": 0.003,
    },
}


def create_flowmatch_scheduler(
    config_name: str = "balanced",
    **override_kwargs
) -> FlowMatchScheduler:
    """
    Factory function to create FlowMatch scheduler from presets.
    
    Args:
        config_name: One of 'quality', 'balanced', 'turbo_fast', 'turbo_quality'
        **override_kwargs: Override specific parameters
        
    Returns:
        Configured FlowMatchScheduler instance
    """
    if config_name not in FLOWMATCH_PRESETS:
        raise ValueError(f"Unknown preset: {config_name}. Options: {list(FLOWMATCH_PRESETS.keys())}")
    
    config = FLOWMATCH_PRESETS[config_name].copy()
    config.update(override_kwargs)
    
    return FlowMatchScheduler(**config)
