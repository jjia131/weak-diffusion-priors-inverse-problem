from PIL import Image
import torch
from pathlib import Path
import yaml
import torchvision.transforms as T
from util.optimizer import *
import random
import numpy as np

from src.image_operator import *

def pil_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    """Convert PIL Image to tensor."""
    transform = T.Compose([
        T.ToTensor(),  # [0, 1]
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
    ])
    return transform(img).unsqueeze(0).to(device)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image."""
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)
    
    # Convert to numpy
    img_np = tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype(np.uint8)
    
    return Image.fromarray(img_np)


def load_config(task_name: str) -> dict:
    """Load task configuration from YAML file."""
    config_path = Path(__file__).parent / 'config' / f'{task_name}.yaml'
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_model_id(dataset_name: str) -> str:
    """Get model ID based on dataset name."""
    model_map = {
        'celeba': 'google/ddpm-ema-celebahq-256',
        'church': 'google/ddpm-ema-church-256',
        'bedroom': 'google/ddpm-ema-bedroom-256'
    }
    
    if dataset_name not in model_map:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(model_map.keys())}")
    
    return model_map[dataset_name]


def setup_forward_operator(config: dict, device: torch.device, H: int = 256, W: int = 256):
    """Setup forward operator and noise model based on config."""
    operator_config = config['measurement']['operator']
    noise_config = config['measurement']['noise']
    
    op_name = operator_config['name']
    
    # Setup noise model
    noise_model = get_noise(noise_config['name'], sigma=noise_config['sigma'])
    
    # Setup forward operator
    mask = None
    mask_3ch = None

    # print operator name and parameters
    # print(f"Setting up forward operator: {op_name} with params: {operator_config}")

    if op_name == 'inpainting':
        forward_op = get_operator('inpainting', device=device)
        
        mask_opt = config['measurement']['mask_opt']
        mask_type = mask_opt['mask_type']
        
        if mask_type == 'random':
            prob_range = mask_opt['mask_prob_range']
            keep_ratio = random.uniform(prob_range[0], prob_range[1])
            mask = MaskGenerator.random_mask(H, W, keep_ratio=keep_ratio).to(device)
        elif mask_type == 'box':
            prob_range = mask_opt['mask_prob_range']
            mask = MaskGenerator.box_mask(H, W, prob_range[0], prob_range[1]).to(device)
        
        mask_3ch = MaskGenerator.expand_mask_channels(mask, 3)
        op_inpainting = True
        
    elif op_name == 'super_resolution':
        scale_factor = operator_config['scale_factor']
        in_shape = tuple(operator_config['in_shape'])
        forward_op = get_operator('super_resolution', 
                                 in_shape=in_shape, 
                                 scale_factor=scale_factor, 
                                 device=device)
        op_inpainting = False
        
    elif op_name == 'gaussian_blur':
        kernel_size = operator_config['kernel_size']
        intensity = operator_config['intensity']
        operator_params = {
            'kernel_size': kernel_size,
            'intensity': intensity,
            'device': device
        }
        forward_op = get_operator('gaussian_blur', **operator_params)
        op_inpainting = False
        
    elif op_name == 'nonlinear_blur':
        opt_yml_path = operator_config['opt_yml_path']
        forward_op = get_operator('nonlinear_blur', opt_yml_path=opt_yml_path, device=device)
        op_inpainting = False
    
    else:
        raise ValueError(f"Unknown operator: {op_name}")
    
    return forward_op, noise_model, mask, mask_3ch, op_inpainting

def build_optimizer(params, name: str, lr: float, beta1: float, beta2: float, radius: float, retraction: str):
    if name == "AdamOnSphere":
        return AdamOnSphere(params, lr=lr, radius=radius, betas=(beta1, beta2), retraction=retraction)
    if name == "Adam":
        return torch.optim.Adam(params, lr=lr, betas=(beta1, beta2))
    if name == "SGD":
        return torch.optim.SGD(params, lr=lr)
    raise ValueError(f"Unknown optimizer: {name}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)