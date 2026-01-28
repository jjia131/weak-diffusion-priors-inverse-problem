import numpy as np
import torch


# beta[t] = beta_start + t * (beta_end - beta_start) / (timesteps - 1)
def linear_beta_schedule(steps, beta_start=0.0001, beta_end=0.02):
    return np.linspace(beta_start, beta_end, steps)


# Used on latent space
def scaled_linear(steps, beta_start=0.00085, beta_end=0.012):
    # return (torch.linspace(beta_start**0.5, beta_end**0.5, steps, dtype=torch.float32) ** 2).detach().numpy()
    return np.linspace(beta_start**0.5, beta_end**0.5, steps, dtype=np.float32) ** 2


# `alpha` = 1 - `beta`, and alpha_bar[t] = Prod_{s=1}^t (alpha[s])
def compute_alphas(beta):
    alpha = 1.0 - beta
    alpha_bar = np.cumprod(alpha)
    return alpha, alpha_bar


# This function simulates the noisy data at a given timestep
def add_noise(x_batch, t_batch, alpha_bar_sqrt, one_minus_alpha_bar_sqrt, device):
    batch_size = x_batch.shape[0]

    # random noise
    noise = torch.randn_like(x_batch, device=device)

    # alpha_bar and (1 - alpha_bar) for the current timestep
    alpha_bar_sqrt_batch = alpha_bar_sqrt[t_batch].to(device)
    one_minus_alpha_bar_sqrt_batch = one_minus_alpha_bar_sqrt[t_batch].to(device)

    alpha_bar_sqrt_batch = alpha_bar_sqrt_batch.view(batch_size, 1, 1, 1)
    one_minus_alpha_bar_sqrt_batch = one_minus_alpha_bar_sqrt_batch.view(batch_size, 1, 1, 1)

    # x_t = sqrt(alpha_bar[t]) * x_0 + sqrt(1 - alpha_bar[t]) * noise
    noisy_x = alpha_bar_sqrt_batch * x_batch + one_minus_alpha_bar_sqrt_batch * noise

    return noisy_x, noise


def load_scheduler(scheduler='linear', steps=1000, beta_start=0.0001, beta_end=0.02):
    # Compute beta and alpha
    if scheduler == 'linear':
        beta = linear_beta_schedule(steps=steps, beta_start=beta_start, beta_end=beta_end)
    elif scheduler == 'scaled_linear':
        beta = scaled_linear(steps, beta_start=beta_start, beta_end=beta_end)
    else:
        raise ValueError(f'Unsupported scheduler {scheduler}')
    alpha, alpha_cumprod = compute_alphas(beta)

    # Convert to tensor
    beta = torch.from_numpy(beta).float()
    alpha = torch.from_numpy(alpha).float()
    alpha_cumprod = torch.from_numpy(alpha_cumprod).float()

    return beta, alpha, alpha_cumprod


def ddim_linear_steps(
    total_steps=1000,
    inference_steps=50,
    shifting=0,
    method='linspace'
):
    if method == 'linspace':
        return torch.tensor(list(reversed(range(0, total_steps, total_steps // inference_steps))), dtype=torch.int32) + shifting
    if method == 'leading':
        ratio = total_steps // inference_steps
        # creates integer timesteps by multiplying by ratio
        # casting to int to avoid issues when num_inference_step is power of 3
        timesteps = (np.arange(0, inference_steps) * ratio).round()[::-1].copy().astype(np.int64)
        timesteps += shifting
        return torch.tensor(timesteps)
    else:
        raise ValueError(f'Not implemented: {method}')
