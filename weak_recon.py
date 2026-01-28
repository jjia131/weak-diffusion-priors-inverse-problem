# # Inpainting on CelebA with Church model using GPU 0
# python weak_recon.py --gpu 0 --task inpainting --dataset celeba --model church --start 0 --end 1
import argparse
from pathlib import Path

from src.model import load_model
from src.DDIM_grad_sample import *
from src.image_operator import *
from util.optimizer import *
from util.early_stop import *
from setup import *

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import pyiqa
    
def process_image(
    ref_img_path,
    model,
    forward_op,
    noise_model,
    mask_3ch,
    device,
    es_recon_save_path,
    ps_recon_save_path,
    measure_save_path,
    op_inpainting,
    loss_lpips,
    loss_psnr,
    loss_ssim,
    iters,
    optimizer_name,
    lr,
    beta1,
    beta2,
    retraction,
    early_stop,
    early_stop_best_k,
    early_stop_ratio,
    process_save_path,
    save_process = False,
    seed = None,
):
    ref_img = Image.open(ref_img_path).convert("RGB")
    ref_tensor = pil_to_tensor(ref_img, device)

    set_seed(seed)
    if op_inpainting:
        masked_image = forward_op.forward(ref_tensor, mask=mask_3ch)
        measurement = noise_model(masked_image)
    else:
        measurement = noise_model(forward_op.forward(ref_tensor)).detach()

    # save measurement as numpy array
    np.save(measure_save_path, measurement.cpu().numpy())

    x_T = torch.randn(1, 3, 256, 256, device=device, requires_grad=True)
    
    target_norm = float((3 * 256 * 256) ** 0.5)
    opt = build_optimizer([x_T], optimizer_name, lr, beta1, beta2, target_norm, retraction)

    criterion = torch.nn.MSELoss().to(device)
    customized_ddim_steps = torch.tensor([666, 333, 0], device=device)

    # Early stopping setup and mask splitting for holdout
    if early_stop == "holdout_simple":
        print("Using simple hold-out early stopping.")
        best_tensor_map = {}
        best_epoch_at_stop = None

        # Split measurement into train/val
        meas_channels = measurement.shape[1]
        meas_h, meas_w = measurement.shape[-2:]
        train_keep_ratio = early_stop_ratio
        if op_inpainting:
            available_mask = mask_3ch[0, 0:1].to(device)  # single channel mask
            random_draw = torch.rand_like(available_mask)
            train_mask = ((random_draw < train_keep_ratio) * available_mask).float()
            val_mask = (available_mask - train_mask).clamp(min=0.0)
        else:
            train_mask = MaskGenerator.random_mask(meas_h, meas_w, keep_ratio=train_keep_ratio).to(device)
            val_mask = 1.0 - train_mask
        train_mask3 = MaskGenerator.expand_mask_channels(train_mask, meas_channels)
        val_mask3 = MaskGenerator.expand_mask_channels(val_mask, meas_channels)
        y_train = (train_mask3 * measurement).detach()
        y_val = (val_mask3 * measurement).detach()
        earlystop = SimpleHoldOutStop(patience=1000, best_k=early_stop_best_k)
    elif early_stop == "variance":
        print("Using variance-based early stopping.")
        earlystop = EarlyStop(size=10, patience=100)
    else:
        raise ValueError(f"Unknown early stop method: {early_stop}")

    best_img = None
    es_best_img = None
    best_psnr = -float("inf")

    if save_process:
        process_save_path = Path(process_save_path)
        process_save_path.mkdir(parents=True, exist_ok=True)
        save_indices = [0, 4, 9, 99, 999]

    pbar = tqdm(range(iters), desc="Weak", ncols=130)
    for iteration in pbar:
        opt.zero_grad(set_to_none=True)

        x_0, _, _ = DDIM_sampling(
            x_t=x_T,
            model=model,
            total_steps=1000,
            inference_steps=3,
            device=device,
            verbose=False,
            enable_grad=True,
            ext_ddim_steps=customized_ddim_steps,
        )

        if early_stop == "holdout_simple":
            if op_inpainting:
                pred_measurement = forward_op.forward(x_0, mask=mask_3ch)
            else:
                pred_measurement = forward_op.forward(x_0)
            loss = criterion(train_mask3 * pred_measurement, y_train)
        else:
            if op_inpainting:
                pred_measurement = forward_op.forward(x_0, mask=mask_3ch)
            else:
                pred_measurement = forward_op.forward(x_0)
            loss = criterion(pred_measurement, measurement)

        loss.backward()
        opt.step()

        with torch.no_grad():
            norm = x_T.view(x_T.size(0), -1).norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
            x_T.mul_((target_norm / norm).view(-1, 1, 1, 1))

            recon_01 = ((x_0 + 1) / 2).clamp(0, 1)
            ref_01 = ((ref_tensor + 1) / 2).clamp(0, 1)

            psnr_value = float(loss_psnr(recon_01, ref_01).cpu().item())

            # ---- save process frame ----
            if save_process and iteration in save_indices:
                out_path = process_save_path / f"iter_{iteration:04d}.npy"
                np.save(out_path, recon_01.detach().cpu().numpy())
            # ----------------------------

            if psnr_value > best_psnr:
                best_psnr = psnr_value
                best_img = x_0.detach()
            
            if early_stop == "holdout_simple":
                val_loss = criterion(val_mask3 * pred_measurement, y_val).item()
                stop_flag, best_epoch = earlystop.update(val_loss, iteration)
                
                if best_epoch is not None and best_epoch == iteration:
                    best_tensor_map[iteration] = x_0.detach().clone()

                if stop_flag:
                    best_epoch_at_stop = earlystop.best_epoch
            else:
                recon_flat = recon_01.detach().cpu().squeeze().numpy().reshape(-1)
                check_early_stop(earlystop=earlystop, new_sample=recon_flat, cur_epoch=iteration)
                stop_flag = earlystop.stop

                if stop_flag and es_best_img is None:
                    es_best_img = x_0.detach()
    
        pbar.set_postfix(
            {   
                "loss": f"{float(loss):.3f}",
                "psnr": f"{psnr_value:.3f}",
            }
        )
    if early_stop == "holdout_simple":
        chosen_epoch = best_epoch_at_stop if best_epoch_at_stop is not None else earlystop.best_epoch
        es_best_img = best_tensor_map.get(chosen_epoch)

        if es_best_img is None and best_tensor_map:
            print(f"Fallback to last best candidate in best_tensor_map. using {max(best_tensor_map.keys())}")
            es_best_img = best_tensor_map[max(best_tensor_map.keys())]

        if es_best_img is None:
            print("Fallback to last iteration x_0 for es_best_img.")
            es_best_img = x_0.detach()  # or best_img
    else:
        if es_best_img is None:
            es_best_img = best_img
    
    if best_img is not None:
        ps_recon_save_path.parent.mkdir(parents=True, exist_ok=True)
        ps_recon = ((best_img.detach() + 1) / 2).clamp(0, 1).cpu().numpy()
        np.save(ps_recon_save_path.with_suffix('.npy'), ps_recon)

    if es_best_img is not None:
        es_recon_save_path.parent.mkdir(parents=True, exist_ok=True)
        es_recon = ((es_best_img.detach() + 1) / 2).clamp(0, 1).cpu().numpy()
        np.save(es_recon_save_path.with_suffix('.npy'), es_recon)

    ssim_value = float(loss_ssim(recon_01, ref_01).cpu().item())
    lpips_value = float(loss_lpips(recon_01, ref_01).cpu().item())

    # print es stop image metric result for debug
    recon_01 = ((es_best_img + 1) / 2).clamp(0, 1)
    ref_01 = ((ref_tensor + 1) / 2).clamp(0, 1)
    print("ES Stop stopped at iteration:", earlystop.best_epoch)
    es_psnr_value = float(loss_psnr(recon_01, ref_01).cpu().item())
    es_ssim_value = float(loss_ssim(recon_01, ref_01).cpu().item())
    es_lpips_value = float(loss_lpips(recon_01, ref_01).cpu().item())

    # calculate ps metrics
    recon_01 = ((best_img + 1) / 2).clamp(0, 1)
    ref_01 = ((ref_tensor + 1) / 2).clamp(0, 1)
    best_psnr = float(loss_psnr(recon_01, ref_01).cpu().item())
    best_ssim = float(loss_ssim(recon_01, ref_01).cpu().item())
    best_lpips = float(loss_lpips(recon_01, ref_01).cpu().item())

    return {
        "stopped_epoch": earlystop.best_epoch if earlystop.best_epoch is not None else 999,
        "final_loss": float(loss.detach().cpu().item()),
        "final_psnr": psnr_value,
        "final_ssim": ssim_value,
        "final_lpips": lpips_value,
        "best_psnr" : best_psnr,
        "best_ssim" : best_ssim,
        "best_lpips" : best_lpips,
        'es_final_psnr': es_psnr_value,
        'es_final_ssim': es_ssim_value,
        'es_final_lpips': es_lpips_value,
    }

def main():
    parser = argparse.ArgumentParser(description="Weak Image Reconstruction")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["inpainting", "gaussian", "super", "nonlinear"],
    )
    parser.add_argument("--dataset", type=str, required=True, choices=["celeba", "church", "bedroom"])
    parser.add_argument("--model", type=str, required=True, choices=["celeba", "church", "bedroom"])
    parser.add_argument("--save_path", type=str, default="weak_result")
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--retraction", type=str, default="normalize", choices=["normalize", "exp"])
    parser.add_argument("--optimizer", type=str, default="AdamOnSphere", choices=["AdamOnSphere", "Adam", "SGD"])
    parser.add_argument("--early_stop", type=str, default="holdout_simple", choices=["variance", "holdout_simple"])
    parser.add_argument("--early_stop_best_k", type=int, default=100)
    parser.add_argument("--early_stop_ratio", type=float, default=0.9)

    args = parser.parse_args()

    # print all args
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    device = torch.device(f"cuda:{args.gpu}")
    print(f"Using device: {device}")

    task_choice_name = {"inpainting" : "inpainting", "gaussian":"gaussian_blur", "super":"super_resolution", "nonlinear":"nonlinear_blur"}
    config = load_config(task_choice_name[args.task])

    model_id = get_model_id(args.model)
    model = load_model(model_id)
    model.to(device)
    model.eval()

    loss_lpips = pyiqa.create_metric("lpips-vgg", device=device)
    loss_psnr = pyiqa.create_metric("psnr", device=device)
    loss_ssim = pyiqa.create_metric("ssimc", device=device)

    data_dir = Path(args.data_path) / args.dataset
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    image_files = sorted(list(data_dir.glob("*.png")))
    image_files = image_files[args.start : args.end]
    print(f"Found {len(image_files)} images in {data_dir}")

    recon_dir = Path(args.save_path) / args.model / args.dataset / args.task / "recon"
    measure_dir = Path(args.save_path) / args.model / args.dataset / args.task / "measure"
    process_dir = Path(args.save_path) / args.model / args.dataset / args.task / "process"
    print(f"Reconstruction will be saved to: {recon_dir}")
    
    recon_dir.mkdir(parents=True, exist_ok=True)
    measure_dir.mkdir(parents=True, exist_ok=True)
    process_dir.mkdir(parents=True, exist_ok=True)

    random_seed = args.seed
        
    for index, img_path in enumerate(image_files):
        
        seed = random_seed + index
        set_seed(seed)

        print(f"\n{'='*60}")
        print(f"Processing: {img_path.name}")
        print(f"{'='*60}")

        forward_op, noise_model, _, mask_3ch, op_inpainting = setup_forward_operator(config, device, H=256, W=256)

        es_recon_save_path = recon_dir / f"{img_path.stem}_es_recon.npy"
        ps_recon_save_path = recon_dir / f"{img_path.stem}_ps_recon.npy"
        measure_save_path = measure_dir / f"{img_path.stem}_measure.npy"
        process_save_path = process_dir / img_path.stem

        metrics = process_image(
            img_path,
            model,
            forward_op,
            noise_model,
            mask_3ch,
            device,
            es_recon_save_path,
            ps_recon_save_path,
            measure_save_path,
            op_inpainting,
            loss_lpips,
            loss_psnr,
            loss_ssim,
            args.iters,
            args.optimizer,
            args.lr,
            args.beta1,
            args.beta2,
            args.retraction,
            args.early_stop,
            args.early_stop_best_k,
            args.early_stop_ratio,
            process_save_path=process_save_path,
            save_process=False,
            seed = seed
        )

        # print(f"\nSaved measurement to: {measure_save_path}")
        print(f"Saved Best reconstruction to: {ps_recon_save_path}")
        print(f"Saved ES reconstruction to: {es_recon_save_path}")
        print(f"Stopped Epoch: {metrics['stopped_epoch']}")
        print(f"Final PSNR:     {metrics['final_psnr']:.2f}")
        print(f"ES PSNR:        {metrics['es_final_psnr']:.2f}")
        print(f"Best PSNR:      {metrics['best_psnr']:.2f}")

        # save metrics to a npz file
        metrics_save_path = recon_dir / f"{img_path.stem}_metrics.npz"
        np.savez(metrics_save_path, **metrics)
        print(f"Saved metrics to: {metrics_save_path}")

if __name__ == "__main__":
    main()
