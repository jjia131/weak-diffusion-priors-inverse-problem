import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.joinpath('src').resolve()))
base_path = Path(__file__).parent.parent.parent.resolve()

import warnings

import torch
from scheduler import ddim_linear_steps, load_scheduler, scaled_linear
from torchvision import transforms
from tqdm import tqdm

def DDIM_sampling(
    x_t,
    model,
    total_steps,
    inference_steps,
    start_step_idx: int = 0,
    end_step_idx: int = -1,
    scheduler_method='linear',
    ddim_method='linear',
    eta=0,
    device=torch.device('cuda:1'),
    inverse=False,
    clip=True,
    verbose=True,
    work_on_latent=False,
    text_embeddings=None,
    do_classifier_free_guidance=True,
    guidance_scale=3.5,
    pipe=None,
    enable_grad=True,
    ext_ddim_steps = None,
):
    model.to(device)
    model.eval()

    # Only cumulative production used in DDIM
    if work_on_latent:
        assert scheduler_method == 'scaled_linear', '"scheduler_method" should be `scaled_linear` when working on latent space'
        assert clip == False, '"clip" should be `False` when working on latent space'
        _, _, alpha_cumprod = load_scheduler(scheduler=scheduler_method, steps=total_steps, beta_start=0.00085, beta_end=0.012)
    else:
        _, _, alpha_cumprod = load_scheduler(scheduler_method)
    alpha_cumprod = alpha_cumprod.to(device)

    if ddim_method == "linear":
        ddim_steps = ddim_linear_steps(total_steps, inference_steps, method='leading', shifting=1).to(device)
    else:
        raise ValueError(f"Unsupported method: {ddim_method}")

    if inverse:
        ddim_steps = reversed(ddim_steps)

    prednoise_per_step = []
    predx0_per_step = []
    latent_hist = []

    if end_step_idx == -1:
        end_step_idx = len(ddim_steps)

    # print(start_step_idx, end_step_idx)
    if ext_ddim_steps is not None:
        ddim_steps = ext_ddim_steps

    # print(ddim_steps)
    grad_ctx = torch.enable_grad() if enable_grad else torch.no_grad()
    with grad_ctx:
        for i in tqdm(range(start_step_idx, end_step_idx), disable=not verbose):
            t = ddim_steps[i]
            alpha_t = alpha_cumprod[t]

            if i == len(ddim_steps) - 1:
                if inverse:
                    continue
                else:
                    prev_t = t
                    if work_on_latent:
                        alpha_t_prev = alpha_cumprod[prev_t]
                    else:
                        alpha_t_prev = torch.tensor(1.0, device=device)
            else:
                prev_t = ddim_steps[i + 1]
                alpha_t_prev = alpha_cumprod[prev_t]

            if i == 0:
                ##### Formula (12) #####
                if text_embeddings is None:
                    predicted_noise = model(x_t, t).sample
                else:
                    if do_classifier_free_guidance:
                        latent = torch.cat([x_t] * 2)
                        noise_pred = model(latent, prev_t if inverse else t, encoder_hidden_states=text_embeddings).sample
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        predicted_noise = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    else:
                        predicted_noise = model(x_t, t, encoder_hidden_states=text_embeddings).sample
    
                predicted_x0 = (x_t - (1.0 - alpha_t).sqrt() * predicted_noise) / alpha_t.sqrt()
    
                if clip:
                    predicted_x0 = predicted_x0.clamp(min=-1.0, max=1.0)
                    predicted_noise = (x_t - alpha_t.sqrt() * predicted_x0) / (1.0 - alpha_t).sqrt()

                direction = (1.0 - alpha_t_prev).sqrt() * predicted_noise
                x_0 = (alpha_t_prev).sqrt() * predicted_x0 + direction
            else:
                ##### Formula (12) #####
                if text_embeddings is None:
                    predicted_noise = model(x_0, t).sample
                else:
                    if do_classifier_free_guidance:
                        latent = torch.cat([x_0] * 2)
                        noise_pred = model(latent, prev_t if inverse else t, encoder_hidden_states=text_embeddings).sample
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        predicted_noise = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    else:
                        predicted_noise = model(x_0, t, encoder_hidden_states=text_embeddings).sample
    
                predicted_x0 = (x_0 - (1.0 - alpha_t).sqrt() * predicted_noise) / alpha_t.sqrt()
    
                if clip:
                    predicted_x0 = predicted_x0.clamp(min=-1.0, max=1.0)
                    predicted_noise = (x_0 - alpha_t.sqrt() * predicted_x0) / (1.0 - alpha_t).sqrt()
    
                direction = (1.0 - alpha_t_prev).sqrt() * predicted_noise
                x_0 = (alpha_t_prev).sqrt() * predicted_x0 + direction
                

    return x_0, prednoise_per_step, predx0_per_step


def one_step(
    x_t,
    model,
    total_steps,
    inference_steps,
    start_step_idx: int = 0,
    end_step_idx: int = -1,
    scheduler_method='linear',
    ddim_method='linear',
    eta=0,
    device=torch.device('cuda:1'),
    inverse=False,
    clip=True,
    verbose=True,
    work_on_latent=False,
    text_embeddings=None,
    do_classifier_free_guidance=True,
    guidance_scale=3.5,
    pipe=None,
    enable_grad=True,
    ext_ddim_steps = None,
):
    model.to(device)
    model.eval()

    # Only cumulative production used in DDIM
    if work_on_latent:
        assert scheduler_method == 'scaled_linear', '"scheduler_method" should be `scaled_linear` when working on latent space'
        assert clip == False, '"clip" should be `False` when working on latent space'
        _, _, alpha_cumprod = load_scheduler(scheduler=scheduler_method, steps=total_steps, beta_start=0.00085, beta_end=0.012)
    else:
        _, _, alpha_cumprod = load_scheduler(scheduler_method)
    alpha_cumprod = alpha_cumprod.to(device)

    if ddim_method == "linear":
        ddim_steps = ddim_linear_steps(total_steps, inference_steps, method='leading', shifting=1).to(device)
    else:
        raise ValueError(f"Unsupported method: {ddim_method}")

    if inverse:
        ddim_steps = reversed(ddim_steps)

    prednoise_per_step = []
    predx0_per_step = []
    latent_hist = []

    if end_step_idx == -1:
        end_step_idx = len(ddim_steps)

    # print(start_step_idx, end_step_idx)
    if ext_ddim_steps is not None:
        ddim_steps = ext_ddim_steps

    # print(ddim_steps)
    grad_ctx = torch.enable_grad() if enable_grad else torch.no_grad()
    with grad_ctx:
        for i in tqdm(range(start_step_idx, end_step_idx), disable=not verbose):
            t = ddim_steps[i]
            alpha_t = alpha_cumprod[t]

            if i == len(ddim_steps) - 1:
                if inverse:
                    continue
                else:
                    prev_t = t
                    if work_on_latent:
                        alpha_t_prev = alpha_cumprod[prev_t]
                    else:
                        alpha_t_prev = torch.tensor(1.0, device=device)
            else:
                prev_t = ddim_steps[i + 1]
                alpha_t_prev = alpha_cumprod[prev_t]

            if i == 0:
                ##### Formula (12) #####
                if text_embeddings is None:
                    predicted_noise = model(x_t, t).sample
                else:
                    if do_classifier_free_guidance:
                        latent = torch.cat([x_t] * 2)
                        noise_pred = model(latent, prev_t if inverse else t, encoder_hidden_states=text_embeddings).sample
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        predicted_noise = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    else:
                        predicted_noise = model(x_t, t, encoder_hidden_states=text_embeddings).sample
    
                predicted_x0 = (x_t - (1.0 - alpha_t).sqrt() * predicted_noise) / alpha_t.sqrt()
    
                if clip:
                    predicted_x0 = predicted_x0.clamp(min=-1.0, max=1.0)
                #     predicted_noise = (x_t - alpha_t.sqrt() * predicted_x0) / (1.0 - alpha_t).sqrt()

                # direction = (1.0 - alpha_t_prev).sqrt() * predicted_noise
                x_0 = predicted_x0
            else:
                ##### Formula (12) #####
                if text_embeddings is None:
                    predicted_noise = model(x_0, t).sample
                else:
                    if do_classifier_free_guidance:
                        latent = torch.cat([x_0] * 2)
                        noise_pred = model(latent, prev_t if inverse else t, encoder_hidden_states=text_embeddings).sample
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        predicted_noise = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    else:
                        predicted_noise = model(x_0, t, encoder_hidden_states=text_embeddings).sample
    
                predicted_x0 = (x_0 - (1.0 - alpha_t).sqrt() * predicted_noise) / alpha_t.sqrt()
    
                if clip:
                    predicted_x0 = predicted_x0.clamp(min=-1.0, max=1.0)
                #     predicted_noise = (x_0 - alpha_t.sqrt() * predicted_x0) / (1.0 - alpha_t).sqrt()
    
                # direction = (1.0 - alpha_t_prev).sqrt() * predicted_noise
                x_0 = predicted_x0
                

    return x_0, prednoise_per_step, predx0_per_step

def latent_wrapper_one_step(
    model,
    prompt,
    prompt_encoder,
    start_latents=None,
    image=None,
    image_encoder=None,
    total_steps=1000,
    inference_steps=50,
    start_step_idx=0,
    end_step_idx=-1,
    inverse=False,
    do_classifier_free_guidance=True,
    num_images_per_prompt=1,
    negative_prompt="",
    device=torch.device('cuda:1'),
    verbose=True,
    guidance_scale=3.5,
    enable_grad = True,
    ext_ddim_steps = None,
):
    text_embeddings = prompt_encoder(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    if inverse:
        if image is not None:
            assert image_encoder is not None, '"image_encoder" should be provided when providing "image"'
            if start_latents is not None:
                warnings.warn('start_latents is ignored by the image')
            latent = image_encoder(transforms.functional.to_tensor(image).unsqueeze(0).to(device) * 2 - 1)
            start_latents = 0.18215 * latent.latent_dist.sample()
            
        assert start_latents is not None, '"start_latents" should not be `None` when inversing'
        if end_step_idx == -1:
            end_step_idx=inference_steps - 2
            # print(end_step_idx)
        
    if start_latents is None:
        start_latents = torch.randn(1, 4, 64, 64, device=device)
        # start_latents *= pipe.scheduler.init_noise_sigma
    
    latents = start_latents.clone()

    latents = one_step(
        x_t=latents,
        model=model,
        total_steps=total_steps,
        inference_steps=inference_steps,
        device=device,
        start_step_idx=start_step_idx,
        end_step_idx=end_step_idx,
        inverse=inverse,
        work_on_latent=True,
        text_embeddings=text_embeddings,
        do_classifier_free_guidance=do_classifier_free_guidance,
        guidance_scale=guidance_scale,
        clip=False,
        scheduler_method='scaled_linear',
        verbose=verbose,
        enable_grad = enable_grad,
        ext_ddim_steps = ext_ddim_steps
    )[0]

    return latents

def latent_wrapper(
    model,
    prompt,
    prompt_encoder,
    start_latents=None,
    image=None,
    image_encoder=None,
    total_steps=1000,
    inference_steps=50,
    start_step_idx=0,
    end_step_idx=-1,
    inverse=False,
    do_classifier_free_guidance=True,
    num_images_per_prompt=1,
    negative_prompt="",
    device=torch.device('cuda:1'),
    verbose=True,
    guidance_scale=3.5,
    enable_grad = True,
    ext_ddim_steps = None,
):
    text_embeddings = prompt_encoder(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    if inverse:
        if image is not None:
            assert image_encoder is not None, '"image_encoder" should be provided when providing "image"'
            if start_latents is not None:
                warnings.warn('start_latents is ignored by the image')
            latent = image_encoder(transforms.functional.to_tensor(image).unsqueeze(0).to(device) * 2 - 1)
            start_latents = 0.18215 * latent.latent_dist.sample()
            
        assert start_latents is not None, '"start_latents" should not be `None` when inversing'
        if end_step_idx == -1:
            end_step_idx=inference_steps - 2
            # print(end_step_idx)
        
    if start_latents is None:
        start_latents = torch.randn(1, 4, 64, 64, device=device)
        # start_latents *= pipe.scheduler.init_noise_sigma
    
    latents = start_latents.clone()

    latents = DDIM_sampling(
        x_t=latents,
        model=model,
        total_steps=total_steps,
        inference_steps=inference_steps,
        device=device,
        start_step_idx=start_step_idx,
        end_step_idx=end_step_idx,
        inverse=inverse,
        work_on_latent=True,
        text_embeddings=text_embeddings,
        do_classifier_free_guidance=do_classifier_free_guidance,
        guidance_scale=guidance_scale,
        clip=False,
        scheduler_method='scaled_linear',
        verbose=verbose,
        enable_grad = enable_grad,
        ext_ddim_steps = ext_ddim_steps
    )[0]

    return latents

def decode_latents(pipe, latents):
    # latents: (B,4,64,64)
    latents = 1 / 0.18215 * latents
    with torch.enable_grad():  # must allow grad to flow if you optimize
        image = pipe.vae.decode(latents).sample  # (B,3,256,256)
    return torch.tanh(image)  # squash to [-1,1] to match ref_tensor

def compute_log_posterior(Z_input, x_0, operator, sigma, measurement, **operator_kwargs):    
    # Apply measurement model with any operator
    y_pred = operator.forward(x_0, **operator_kwargs)
    
    recon_error = ((y_pred - measurement) ** 2).sum()
    log_likelihood = -0.5 / (sigma**2) * recon_error
    log_prior = -0.5 * (Z_input ** 2).sum()
    
    # Log posterior
    log_posterior = log_likelihood + log_prior
    
    return log_posterior