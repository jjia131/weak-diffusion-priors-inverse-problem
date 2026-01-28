# # Inpainting on CelebA with GPU 0
# python dps_recon.py --gpu 0 --task inpainting --dataset celeba --model celeba --start 0 --end 1
import argparse
from pathlib import Path

from src.model import load_model
from src.DDIM_grad_sample import *
from src.image_operator import *
from src.scheduler import load_scheduler
from setup import *

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import pyiqa

def ddpm_step(model_output, img, i, ddpm_steps, beta, alpha, alpha_cumprod, device):
    """Perform one DDPM denoising step."""
    curr_step = ddpm_steps[i]
    beta_curr = beta[curr_step]
    alpha_curr = alpha[curr_step]
    alpha_cumprod_curr = alpha_cumprod[curr_step]

    if i == len(ddpm_steps) - 1:
        alpha_cumprod_prev = torch.tensor(1.0, device=device)
    else:
        alpha_cumprod_prev = alpha_cumprod[curr_step - 1]

    ##### Formula (15) #####
    predicted_x0 = (img - torch.sqrt(1.0 - alpha_cumprod_curr) * model_output) / torch.sqrt(
        alpha_cumprod_curr)
    predicted_x0 = predicted_x0.clamp(min=-1.0, max=1.0)

    ##### Formula (7) ######
    img = (torch.sqrt(alpha_cumprod_prev) * beta_curr) / (
                1 - alpha_cumprod_curr) * predicted_x0 + (
                                torch.sqrt(alpha_curr) * (1 - alpha_cumprod_prev)) / (
                                1 - alpha_cumprod_curr) * img
    ########################

    sigma = torch.sqrt(beta_curr)
    if i != len(ddpm_steps) - 1:
        rand_noise = torch.randn_like(img, device=device)
        img += rand_noise * sigma

    return img, predicted_x0


def process_image(ref_img_path, model, forward_op, noise_model, mask_3ch, 
                  config, device, recon_save_path, measurement_save_path,
                  beta, alpha, alpha_cumprod, ddpm_steps, op_inpainting,
                  loss_lpips, loss_psnr, loss_ssim, seed):
    """Process a single image."""
    
    H, W = 256, 256
    
    # Load and preprocess image
    ref_img = Image.open(ref_img_path).convert("RGB")
    ref_tensor = pil_to_tensor(ref_img, device)
    
    set_seed(seed)
    # Create measurement
    if op_inpainting:
        masked_image = forward_op.forward(ref_tensor, mask=mask_3ch)
        measurement = noise_model(masked_image)
    else:
        measurement = noise_model(forward_op.forward(ref_tensor)).detach()

    # save measurement numpy as npy 
    measurement_np = measurement.cpu().numpy()
    measurement_save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(measurement_save_path, measurement_np)
    
    # Get algorithm parameters
    algo_config = config.get('dps_algo', {})
    scale = algo_config.get('scale', 0.5)
    num_inference_steps = 1000
    img = torch.randn_like(ref_tensor).to(device)
    
    ref_01 = ((ref_tensor + 1) / 2).clamp(0, 1)
    
    pbar = tqdm(list(range(num_inference_steps)), 
                desc="DPS", ncols=130)
    
    for idx in pbar:
        img = img.requires_grad_()
        model_output = model(img, ddpm_steps[idx]).sample
        
        pred_x_t, pred_x_start = ddpm_step(model_output, img, idx, ddpm_steps, 
                                           beta, alpha, alpha_cumprod, device)
        
        # Compute gradient
        if op_inpainting:
            difference = measurement - forward_op.forward(pred_x_start, mask=mask_3ch)
        else:
            difference = measurement - forward_op.forward(pred_x_start)
        
        norm = torch.linalg.norm(difference)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=img)[0]
        
        pred_x_t -= norm_grad * scale
        img = pred_x_t.detach_()
        
        pbar.set_postfix({'distance': norm.item()}, refresh=False)

    recon_01 = ((img + 1) / 2).clamp(0, 1)
    psnr_value = loss_psnr(recon_01, ref_01).cpu().item()
    ssim_value = loss_ssim(recon_01, ref_01).cpu().item()
    lpips_value = loss_lpips(recon_01, ref_01).cpu().item()
    
    # save reconstruction numpy as npy 
    recon_np = recon_01.cpu().numpy()
    recon_save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(recon_save_path, recon_np)

    # Save metrics
    metrics = {
        'final_psnr': psnr_value,
        'final_ssim': ssim_value,
        'final_lpips': lpips_value,
    }
    return metrics

def main():
    parser = argparse.ArgumentParser(description='DPS Image Reconstruction')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID (0-7)')
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["inpainting", "gaussian", "super", "nonlinear"],
    )
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['celeba', 'church', 'bedroom'],
                       help='Dataset name')
    parser.add_argument('--model', type=str, required=True,
                       choices=['celeba', 'church', 'bedroom'],
                       help='Model name')
    parser.add_argument('--save_path', type=str, default='dps_result',
                       help='Base path for saving results')
    parser.add_argument('--data_path', type=str, default='data',
                       help='Base path for data')
    parser.add_argument('--start', type=int, default=0,
                       help='start index for data')
    parser.add_argument('--end', type=int, default=100,
                       help='end index for data')
    parser.add_argument('--seed', type=int, default=42,
                       help='random index')    
    
    args = parser.parse_args()

    # print all args
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    # Setup device
    device = torch.device(f"cuda:{args.gpu}")
    print(f"Using device: {device}")
    
    # Load config
    task_choice_name = {"inpainting" : "inpainting", "gaussian":"gaussian_blur", "super":"super_resolution", "nonlinear":"nonlinear_blur"}
    print(f"Loading config for task: {args.task}")
    # config = load_config(args.task)
    config = load_config(task_choice_name[args.task])
    
    # Load model
    model_id = get_model_id(args.model)
    print(f"Loading model: {model_id}")
    model = load_model(model_id)
    model.to(device)
    model.eval()
    print("Model loaded.")

    # Initialize metrics
    loss_lpips = pyiqa.create_metric('lpips-vgg', device=device)
    loss_psnr = pyiqa.create_metric('psnr', device=device)
    loss_ssim = pyiqa.create_metric('ssimc', device=device)
    
    # Load scheduler
    ddpm_steps = torch.tensor(list(reversed(range(1000))), dtype=torch.int32, device=device)
    beta, alpha, alpha_cumprod = load_scheduler("linear")
    beta = beta.to(device)
    alpha = alpha.to(device)
    alpha_cumprod = alpha_cumprod.to(device)
    
    # Get all images in dataset
    data_dir = Path(args.data_path) / args.dataset
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    
    image_files = sorted(list(data_dir.glob("*.png")))
    print(f"Found {len(image_files)} images in {data_dir}")
    
    # Setup save directory
    recon_dir = Path(args.save_path) / args.model / args.dataset / args.task / 'recon'
    measure_dir = Path(args.save_path) / args.model  / args.dataset / args.task / 'measure'
    recon_dir.mkdir(parents=True, exist_ok=True) 
    measure_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reconstructions will be saved to: {recon_dir}")
    print(f"Measurements will be saved to: {measure_dir}")
    
    # Process each image
    all_metrics = {}
    
    random_seed = args.seed    
    for index,img_path in enumerate(image_files[args.start:args.end]):
        seed = random_seed + index
        set_seed(seed)
        
        print(f"\n{'='*60}")
        print(f"Processing: {img_path.name}")
        print(f"{'='*60}")
        
        # Setup forward operator (new mask for each image for inpainting)
        forward_op, noise_model, mask, mask_3ch, op_inpainting = setup_forward_operator(
            config, device, H=256, W=256
        )
        
        # Process image
        recon_save_path = recon_dir / f"{img_path.stem}_recon.npy"
        measure_save_path = measure_dir / f"{img_path.stem}_measure.npy"
        metrics = process_image(
            img_path, model, forward_op, noise_model, mask_3ch,
            config, device, recon_save_path, measure_save_path,
            beta, alpha, alpha_cumprod, ddpm_steps, op_inpainting,
            loss_lpips, loss_psnr, loss_ssim, seed
        )
        
        all_metrics[img_path.stem] = metrics
        
        print(f"\nSaved reconstruction to: {recon_save_path}")
        print(f"Final PSNR: {metrics['final_psnr']:.2f}")
        print(f"Final SSIM: {metrics['final_ssim']:.4f}")
        print(f"Final LPIPS: {metrics['final_lpips']:.4f}")

        # save metrics to npz 
        metrics_path = recon_dir / f"{img_path.stem}_metrics.npz"
        np.savez(metrics_path, **metrics)
        print(f"Saved metrics to: {metrics_path}")

if __name__ == '__main__':
    main()