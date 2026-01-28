from torch.optim import Optimizer
import torch

class AdamOnSphere(Optimizer):
    r"""Adam with a spherical constraint ||x||_2 = radius.

    If radius=None, the radius is taken as the current norm at the first step and then enforced.
    Retraction: 'normalize' (default) or 'exp' (exponential map on the sphere).
    """

    def __init__(self, params, lr=1e-2, betas=(0.9, 0.999), eps=1e-8,
                 radius=None, retraction='normalize'):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        radius=radius, retraction=retraction)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']
            retraction = group['retraction']
            radius_cfg = group['radius']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                x = p.data

                # pick radius: fixed (if provided) or lock to current norm
                r = (torch.as_tensor(radius_cfg, dtype=x.dtype, device=x.device)
                     if radius_cfg is not None else x.norm())
                r = torch.clamp(r, min=1e-12)
                r2 = r * r

                # project gradient to tangent: g_T = g - ((x⋅g)/r^2) x
                xdotg = torch.sum(x * grad)
                g_T = grad - (xdotg / r2) * x

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(x)
                    state['exp_avg_sq'] = torch.zeros_like(x)

                state['step'] += 1
                t = state['step']
                m = state['exp_avg']
                v = state['exp_avg_sq']

                # Adam moments on the tangent gradient
                m.mul_(beta1).add_(g_T, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g_T, g_T, value=1 - beta2)

                # bias correction
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                m_hat = m / bias_correction1
                v_hat = v / bias_correction2

                # preconditioned direction
                u = m_hat / (torch.sqrt(v_hat) + eps)

                # project u back to tangent (important)
                xdotu = torch.sum(x * u)
                p_dir = u - (xdotu / r2) * x

                # take a step and retract to the sphere
                if retraction == 'normalize':
                    x_new = x - lr * p_dir
                    x_new_norm = torch.norm(x_new).clamp_min(1e-12)
                    x.copy_(x_new * (r / x_new_norm))
                elif retraction == 'exp':
                    p_norm = torch.norm(p_dir).clamp_min(1e-12)
                    alpha = lr * (p_norm / r)
                    cos = torch.cos(alpha)
                    sin = torch.sin(alpha)
                    x_unit = x / r
                    v_unit = p_dir / p_norm
                    x.copy_(r * (cos * x_unit + sin * v_unit))
                else:
                    raise ValueError("retraction must be 'normalize' or 'exp'")

        return loss


def compute_log_posterior(Z_input, x_0, operator, sigma, measurement, **operator_kwargs):    
    # Apply measurement model with any operator
    y_pred = operator.forward(x_0, **operator_kwargs)
    
    recon_error = ((y_pred - measurement) ** 2).sum()
    log_likelihood = -0.5 / (sigma**2) * recon_error
    log_prior = -0.5 * (Z_input ** 2).sum()
    
    # Log posterior
    log_posterior = log_likelihood + log_prior
    
    return log_posterior