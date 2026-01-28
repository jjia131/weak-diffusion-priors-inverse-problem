# python main.py --data ./ImageNet --out stable_recon/dsg/inpainting --scale 4.8 --algo dps --operator inpainting --nstep 500 
# python main.py --data ./ImageNet --out stable_recon/dsg/inpainting --scale 0.02 --algo dsg --operator inpainting --nstep 500 
# python main.py --data ./ImageNet --out stable_recon/fdm/inpainting --scale 1.2 --algo fdm --operator inpainting --nstep 500 

# python main.py --data ./ImageNet --out stable_recon/dps/gaussian --scale 4.8 --algo dps --operator gaussian --nstep 500 
# python main.py --data ./ImageNet --out stable_recon/dsg/gaussian --scale 0.02 --algo dsg --operator gaussian --nstep 500 
# python main.py --data ./ImageNet --out stable_recon/fdm/gaussian --scale 1.2 --algo fdm --operator gaussian --nstep 500 

import os
import torch
import random
from ddps.pipe import StableDiffusionInverse, EulerAncestralDSG
from ddps.dataset import ImageDataset
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from torchvision import transforms
import numpy as np
import argparse
from setup import *
import pyiqa

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="diffusers-DPS")
    parser.add_argument(
        "--model",
        type=str,
        default="Manojb/stable-diffusion-2-1-base",
        help="base diffusion model",
    )
    parser.add_argument("--data", type=str, help="path to image folder")
    parser.add_argument("--out", type=str, help="path to output folder")
    parser.add_argument("--scale", type=float, default=4.8, help="scale of DPS")
    parser.add_argument("--prompt", type=str, default="", help="prompt")
    parser.add_argument("--algo", type=str, default="dps", help="algorithm to use")
    parser.add_argument("--operator", type=str, default="inpainting", help="operator to use")
    parser.add_argument("--nstep", type=int, default=500, help="num of steps")
    parser.add_argument("--ngpu", type=int, default=1, help="num of gpu")
    parser.add_argument("--rank", type=int, default=0, help="local rank")

    # FreeDOM specific parameters
    # repeat for K steps, in time interval [c1, c2]
    parser.add_argument("--fdm_c1", type=int, default=100, help="c1 of FreeDOM")
    parser.add_argument("--fdm_c2", type=int, default=250, help="c2 of FreeDOM")
    parser.add_argument("--fdm_k", type=int, default=2, help="k of FreeDOM")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id to use")

    # PSLD specific parameters
    parser.add_argument("--psld_gamma", type=float, default=0.1, help="gamma of PSLD")

    args = parser.parse_args()

    DTYPE = torch.float32

    # out_dirs = ["source", "measure", "recon", "recon_low_res"]
    out_dirs = ["measure", "recon"]
    out_dirs = [os.path.join(args.out, o) for o in out_dirs]
    for out_dir in out_dirs:
        os.makedirs(out_dir, exist_ok=True)

    test_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset = ImageDataset(root=args.data, transform=test_transforms, return_path=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    task_choice_name = {"inpainting" : "inpainting", "gaussian":"gaussian_blur"}
    config = load_config(task_choice_name[args.operator])

    # f = f.to(dtype=DTYPE, device="cuda")

    device = torch.device(f"cuda:{args.gpu_id}")

    loss_lpips = pyiqa.create_metric("lpips-vgg", device=device)
    loss_psnr = pyiqa.create_metric("psnr", device=device)
    loss_ssim = pyiqa.create_metric("ssimc", device=device)
    random_seed = args.seed
    model_id = args.model
    if args.algo == "dsg":
        scheduler = EulerAncestralDSG.from_pretrained(model_id, subfolder="scheduler")
    else:
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
    pipe = StableDiffusionInverse.from_pretrained(
        model_id, scheduler=scheduler, torch_dtype=DTYPE
    )
    pipe = pipe.to(device)
    print("Loaded {} images from {}".format(len(dataset), args.data))
    for i, (x, x_path) in enumerate(dataloader):
        
        # skip for multi gpu

        if i % args.ngpu != args.rank:
            continue
        seed = random_seed + i
        fix_seed(seed)
        x_name = x_path[0].split("/")[-1]
        x_name = x_name[:-4] + ".png"
        x = x.to(dtype=DTYPE, device=device)
        x = torch.nn.functional.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)

        forward_op, noise_model, _, mask_3ch, op_inpainting = setup_forward_operator(config, device, H=512, W=512)
        if op_inpainting:
            masked_image = forward_op.forward(x, mask=mask_3ch)
            y = noise_model(masked_image)
        else:
            y = noise_model(forward_op.forward(x)).detach()

        image, _ = pipe(
            op_inpainting = op_inpainting,
            forward_op = forward_op,
            mask_3ch = mask_3ch,
            y=y,
            algo=args.algo,
            scale=args.scale,
            prompt=args.prompt,
            height=512,
            width=512,
            num_inference_steps=args.nstep,
            guidance_scale=0.0,
            output_type="pt",
            return_dict=False,
            fdm_c1=args.fdm_c1,
            fdm_c2=args.fdm_c2,
            fdm_k=args.fdm_k,
            psld_gamma=args.psld_gamma,
        )
        x_hat = image * 2.0 - 1.0
        if op_inpainting:
            y_hat = forward_op.forward(x_hat, mask=mask_3ch)
        else:
            y_hat = forward_op.forward(x_hat)

        out_tensors = [x, y, x_hat, y_hat]
        x_hat_01 = (x_hat + 1.0) / 2.0
        x_01 = (x + 1.0) / 2.0
        psnr_value = loss_psnr(x_hat_01, x_01).item()
        ssim_value = loss_ssim(x_hat_01, x_01).item()
        lpips_value = loss_lpips(x_hat_01, x_01).item()
        print(f"[{i+1}/{len(dataloader)}] {x_name} PSNR: {psnr_value:.4f} dB, SSIM: {ssim_value:.4f}, LPIPS: {lpips_value:.4f}")

        # save recon as npy 
        np.save(os.path.join(args.out, "recon", x_name[:-4] + "npy"), x_hat.squeeze().cpu().numpy())
        print("Saved image: {}".format(x_name[:-4] + "npy"))