"""
Image Operators Module
======================
This module provides a comprehensive framework for image transformations including:
- Linear operators (blur, super-resolution, inpainting)
- Nonlinear operators (phase retrieval, nonlinear blur)
- Noise models (Gaussian, Poisson)
- Mask generators (box, random, freeform)

The module simulates measurement processes: y = A(x) + n
where A is an operator and n is noise.
"""

import random
from abc import ABC, abstractmethod
from functools import partial
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

# Optional imports - comment out if not available
try:
    from motionblur.motionblur import Kernel
    HAS_MOTIONBLUR = True
except ImportError:
    HAS_MOTIONBLUR = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Utility imports (you'll need to ensure these are available)
from util.resizer import Resizer
from util.img_utils import Blurkernel, fft2_m, perform_tilt

# ==================
# Registry Pattern
# ==================

class OperatorRegistry:
    """Registry for managing operators."""
    def __init__(self):
        self._registry = {}
    
    def register(self, name: str):
        """Decorator to register an operator."""
        def wrapper(cls):
            if name in self._registry:
                raise NameError(f"Operator '{name}' is already registered!")
            self._registry[name] = cls
            return cls
        return wrapper
    
    def get(self, name: str, **kwargs):
        """Get an operator instance by name."""
        if name not in self._registry:
            raise NameError(f"Operator '{name}' is not defined.")
        return self._registry[name](**kwargs)
    
    def list_operators(self):
        """List all registered operators."""
        return list(self._registry.keys())


# Global registries
OPERATOR_REGISTRY = OperatorRegistry()
NOISE_REGISTRY = OperatorRegistry()


# ==================
# Base Classes
# ==================

class LinearOperator(ABC):
    """Base class for linear operators: y = Ax"""
    
    @abstractmethod
    def forward(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply operator: A * x"""
        pass
    
    @abstractmethod
    def transpose(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply transpose: A^T * x"""
        pass
    
    def ortho_project(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        """Orthogonal projection: (I - A^T * A) * x"""
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)
    
    def project(self, data: torch.Tensor, measurement: torch.Tensor, **kwargs) -> torch.Tensor:
        """Project: (I - A^T * A) * y - A * x"""
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


class NonLinearOperator(ABC):
    """Base class for nonlinear operators."""
    
    @abstractmethod
    def forward(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply nonlinear operator."""
        pass
    
    def project(self, data: torch.Tensor, measurement: torch.Tensor, **kwargs) -> torch.Tensor:
        """Nonlinear projection."""
        return data + measurement - self.forward(data, **kwargs)


class Noise(ABC):
    """Base class for noise models."""
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Add noise to data."""
        pass


# ==================
# Mask Generators
# ==================

class MaskGenerator:
    """Collection of mask generation methods for inpainting."""
    
    @staticmethod
    def box_mask(H: int, W: int, min_frac: float = 0.25, max_frac: float = 0.6) -> torch.Tensor:
        """
        Generate a random box mask (0 inside box, 1 outside).
        
        Args:
            H, W: Image dimensions
            min_frac, max_frac: Min/max fraction of image size for box dimensions
        
        Returns:
            Mask tensor of shape (1, 1, H, W)
        """
        M = torch.ones(1, 1, H, W)
        box_h = int(random.uniform(min_frac, max_frac) * H)
        box_w = int(random.uniform(min_frac, max_frac) * W)
        top = random.randint(0, H - box_h)
        left = random.randint(0, W - box_w)
        M[:, :, top:top+box_h, left:left+box_w] = 0.0
        return M
    
    @staticmethod
    def random_mask(H: int, W: int, keep_ratio: float = 0.5) -> torch.Tensor:
        """
        Generate a random binary mask.
        
        Args:
            H, W: Image dimensions
            keep_ratio: Probability of keeping each pixel
        
        Returns:
            Binary mask tensor of shape (1, 1, H, W)
        """
        M = (torch.rand(1, 1, H, W) < keep_ratio).float()
        return M
    
    @staticmethod
    def freeform_mask(H: int, W: int, num_strokes: int = 6, 
                     max_len: int = 80, brush_width: int = 15) -> torch.Tensor:
        """
        Generate free-form stroke mask with smooth brushstrokes.
        
        Args:
            H, W: Image dimensions
            num_strokes: Number of random strokes
            max_len: Maximum length of each stroke
            brush_width: Width of the brush
        
        Returns:
            Binary mask tensor of shape (1, 1, H, W)
        """
        M = torch.ones(1, 1, H, W)
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        yy, xx = yy.float(), xx.float()
        
        for _ in range(num_strokes):
            y, x = random.randint(0, H-1), random.randint(0, W-1)
            theta = random.random() * 2 * np.pi
            length = random.randint(10, max_len)
            dy = torch.sin(torch.tensor(theta))
            dx = torch.cos(torch.tensor(theta))
            
            for t in range(length):
                cy = int(y + dy.item() * t)
                cx = int(x + dx.item() * t)
                if 0 <= cy < H and 0 <= cx < W:
                    # Draw soft brush disk
                    dist2 = (yy - cy)**2 + (xx - cx)**2
                    stroke = torch.exp(-dist2 / (2 * brush_width**2))
                    M = torch.minimum(M, 1.0 - stroke[None, None])
        
        M.clamp_(0, 1)
        return (M < 0.5).float()
    
    @staticmethod
    def expand_mask_channels(mask: torch.Tensor, num_channels: int = 3) -> torch.Tensor:
        """Expand mask from (B, 1, H, W) to (B, C, H, W)."""
        return mask.repeat(1, num_channels, 1, 1)


# ==================
# Linear Operators
# ==================

@OPERATOR_REGISTRY.register('identity')
class IdentityOperator(LinearOperator):
    """Identity operator (no transformation)."""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cpu')
    
    def forward(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        return data
    
    def transpose(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        return data
    
    def ortho_project(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.zeros_like(data)


@OPERATOR_REGISTRY.register('super_resolution')
class SuperResolutionOperator(LinearOperator):
    """Downsampling/upsampling operator for super-resolution."""
    def __init__(self, in_shape: Tuple[int, ...], scale_factor: float, 
                 device: torch.device = None):
        self.device = device or torch.device('cpu')
        self.scale_factor = scale_factor
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)
        self.down_sample = Resizer(in_shape, 1/scale_factor).to(self.device)
    
    def forward(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        """Downsample the image."""
        # Handle batch dimension
        if data.dim() == 4:  # (B, C, H, W)
            batch_size = data.shape[0]
            results = []
            for i in range(batch_size):
                # Process each image in batch separately
                img = data[i]  # (C, H, W)
                downsampled = self.down_sample(img)
                results.append(downsampled.unsqueeze(0))
            return torch.cat(results, dim=0)
        else:  # (C, H, W)
            return self.down_sample(data)
    
    def transpose(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        """Upsample the image."""
        return self.up_sample(data)
    
    def project(self, data: torch.Tensor, measurement: torch.Tensor, **kwargs) -> torch.Tensor:
        """Special projection for super-resolution."""
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)


@OPERATOR_REGISTRY.register('gaussian_blur')
class GaussianBlurOperator(LinearOperator):
    """Gaussian blur operator."""
    
    def __init__(self, kernel_size: int, intensity: float, device: torch.device = None):
        self.device = device or torch.device('cpu')
        self.kernel_size = kernel_size
        self.intensity = intensity
        self.conv = Blurkernel(
            blur_type='gaussian',
            kernel_size=kernel_size,
            std=intensity,
            device=self.device
        ).to(self.device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))
    
    def forward(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply Gaussian blur."""
        return self.conv(data)
    
    def transpose(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        """For blur, transpose is identity."""
        return data
    
    def get_kernel(self) -> torch.Tensor:
        """Get the blur kernel."""
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)


@OPERATOR_REGISTRY.register('inpainting')
class InpaintingOperator(LinearOperator):
    """Inpainting operator using masks."""
    
    def __init__(self, device: torch.device = None, default_mask_type: str = 'box'):
        self.device = device or torch.device('cpu')
        self.default_mask_type = default_mask_type
        self.mask_generator = MaskGenerator()
    
    def forward(self, data: torch.Tensor, mask: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """Apply mask to data."""
        if mask is None:
            # Generate default mask if none provided
            H, W = data.shape[-2:]
            if self.default_mask_type == 'box':
                mask = self.mask_generator.box_mask(H, W)
            elif self.default_mask_type == 'random':
                mask = self.mask_generator.random_mask(H, W)
            else:
                mask = self.mask_generator.freeform_mask(H, W)
            
            # Expand to match channels
            mask = self.mask_generator.expand_mask_channels(mask, data.shape[1])
        
        mask = mask.to(self.device)
        return data * mask
    
    def transpose(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        """For inpainting, transpose is identity."""
        return data
    
    def ortho_project(self, data: torch.Tensor, mask: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """Project onto complement of mask."""
        return data - self.forward(data, mask=mask, **kwargs)


# ==================
# Nonlinear Operators
# ==================

@OPERATOR_REGISTRY.register('phase_retrieval')
class PhaseRetrievalOperator(NonLinearOperator):
    """Phase retrieval operator (Fourier magnitude)."""
    
    def __init__(self, oversample: float = 2.0, device: torch.device = None):
        self.device = device or torch.device('cpu')
        self.pad = int((oversample / 8.0) * 256)
    
    def forward(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute Fourier magnitude."""
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        amplitude = fft2_m(padded).abs()
        return amplitude

@OPERATOR_REGISTRY.register('nonlinear_blur')
class NonlinearBlurOperator(NonLinearOperator):
    def __init__(self, opt_yml_path, device):
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)     
         
    def prepare_nonlinear_blur_model(self, opt_yml_path):
        '''
        Nonlinear deblur requires external codes (bkse).
        '''
        from bkse.models.kernel_encoding.kernel_wizard import KernelWizard
        
        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        blur_model = KernelWizard(opt)
        blur_model.eval()
        blur_model.load_state_dict(torch.load(model_path)) 
        blur_model = blur_model.to(self.device)
        return blur_model
    
    def forward(self, data, **kwargs):
        random_kernel = torch.randn(1, 512, 2, 2).to(self.device) * 1.2
        data = (data + 1.0) / 2.0  #[-1, 1] -> [0, 1]
        blurred = self.blur_model.adaptKernel(data, kernel=random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1) #[0, 1] -> [-1, 1]
        return blurred

# ==================
# Noise Models
# ==================

@NOISE_REGISTRY.register('clean')
class CleanNoise(Noise):
    """No noise (identity)."""
    
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return data


@NOISE_REGISTRY.register('gaussian')
class GaussianNoise(Noise):
    """Additive Gaussian noise."""
    
    def __init__(self, sigma: float = 0.1):
        self.sigma = sigma
    
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(data, device=data.device) * self.sigma
        return data + noise


@NOISE_REGISTRY.register('poisson')
class PoissonNoise(Noise):
    """Poisson (shot) noise."""
    
    def __init__(self, rate: float = 1.0):
        self.rate = rate
    
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # Normalize to [0, 1] for Poisson process
        data_norm = (data + 1.0) / 2.0
        data_norm = data_norm.clamp(0, 1)
        
        # Apply Poisson noise
        device = data.device
        data_cpu = data_norm.detach().cpu()
        noisy = np.random.poisson(data_cpu * 255.0 * self.rate) / (255.0 * self.rate)
        noisy = torch.from_numpy(noisy).float()
        
        # Denormalize back to [-1, 1]
        noisy = noisy * 2.0 - 1.0
        return noisy.clamp(-1, 1).to(device)


# ==================
# Composite Operations
# ==================

class MeasurementProcess:
    """
    Composite measurement process: y = A(x) + n
    Combines an operator with noise and optional masking.
    """
    
    def __init__(self, operator_name: str = 'identity', 
                 noise_name: str = 'clean',
                 operator_params: Dict[str, Any] = None,
                 noise_params: Dict[str, Any] = None,
                 device: torch.device = None):
        
        self.device = device or torch.device('cpu')
        
        # Initialize operator
        operator_params = operator_params or {}
        if 'device' not in operator_params:
            operator_params['device'] = self.device
        self.operator = OPERATOR_REGISTRY.get(operator_name, **operator_params)
        
        # Initialize noise
        noise_params = noise_params or {}
        self.noise = NOISE_REGISTRY.get(noise_name, **noise_params)
    
    def forward(self, x: torch.Tensor, add_noise: bool = True, **kwargs) -> torch.Tensor:
        """Apply measurement process."""
        # Apply operator
        y = self.operator.forward(x, **kwargs)
        
        # Add noise if requested
        if add_noise:
            y = self.noise(y)
        
        return y
    
    def apply_with_mask(self, x: torch.Tensor, mask: torch.Tensor = None, 
                       sigma: float = 0.1) -> torch.Tensor:
        """Apply masking and noise (convenience method)."""
        if mask is not None:
            # Ensure mask has correct shape
            if mask.dim() == 3:
                mask = mask.unsqueeze(0)
            if mask.shape[1] == 1 and x.shape[1] > 1:
                mask = mask.repeat(1, x.shape[1], 1, 1)
            
            x_masked = x * mask.to(x.device)
        else:
            x_masked = x
        
        # Add noise
        noise = torch.randn_like(x_masked) * sigma
        return x_masked + noise


# ==================
# Utility Functions
# ==================

def get_operator(name: str, **kwargs) -> LinearOperator:
    """Get an operator instance by name."""
    return OPERATOR_REGISTRY.get(name, **kwargs)


def get_noise(name: str, **kwargs) -> Noise:
    """Get a noise model instance by name."""
    return NOISE_REGISTRY.get(name, **kwargs)


def list_available_operators() -> list:
    """List all available operators."""
    return OPERATOR_REGISTRY.list_operators()


def list_available_noise_models() -> list:
    """List all available noise models."""
    return NOISE_REGISTRY.list_operators()