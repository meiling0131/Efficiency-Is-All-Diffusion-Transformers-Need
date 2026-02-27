"""
Gaussian Diffusion utilities for training and sampling.
Supports DDPM/DDIM sampling with classifier-free guidance.
"""

import torch


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape):
    """Extract per-timestep values and reshape for broadcasting."""
    out = a.gather(0, t)
    return out.reshape((t.shape[0],) + (1,) * (len(x_shape) - 1))


class GaussianDiffusion:
    def __init__(self, T=1000, device="cpu", beta_start=1e-4, beta_end=2e-2):
        self.T = T
        self.device = torch.device(device)

        betas = torch.linspace(beta_start, beta_end, T, device=self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=self.device), alphas_cumprod[:-1]], dim=0
        )

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # Posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = _extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_omb = _extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        x_t = sqrt_ab * x0 + sqrt_omb * noise
        return x_t, noise

    def _guided_eps(self, model, x_t, t, labels, guidance_scale=3.0):
        num_embeddings = model.class_emb.num_embeddings
        uncond_label = num_embeddings - 1
        uncond = torch.full_like(labels, uncond_label)

        eps_cond = model(x_t, t, labels)
        eps_uncond = model(x_t, t, uncond)
        return eps_uncond + guidance_scale * (eps_cond - eps_uncond)

    @torch.no_grad()
    def ddpm_sample(self, model, shape, labels, guidance_scale=3.0, x_T=None):
        device = self.device
        x = x_T.to(device) if x_T is not None else torch.randn(shape, device=device)

        for t_int in reversed(range(self.T)):
            t = torch.full((shape[0],), t_int, device=device, dtype=torch.long)
            eps = self._guided_eps(model, x, t, labels, guidance_scale)

            alpha_t = _extract(self.alphas, t, x.shape)
            alpha_bar_t = _extract(self.alphas_cumprod, t, x.shape)
            beta_t = _extract(self.betas, t, x.shape)

            # Mean of p(x_{t-1} | x_t)
            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps
            )

            if t_int > 0:
                noise = torch.randn_like(x)
                var = _extract(self.posterior_variance, t, x.shape).clamp(min=1e-20)
                x = mean + torch.sqrt(var) * noise
            else:
                x = mean

        return x.clamp(-1, 1)

    @torch.no_grad()
    def ddim_sample(
        self, model, shape, labels, steps=50, guidance_scale=3.0, x_T=None
    ):
        device = self.device
        x = x_T.to(device) if x_T is not None else torch.randn(shape, device=device)

        times = torch.linspace(self.T - 1, 0, steps, device=device).long()
        prev_times = torch.cat([times[1:], torch.tensor([-1], device=device)])

        for t_int, prev_t_int in zip(times.tolist(), prev_times.tolist()):
            t = torch.full((shape[0],), t_int, device=device, dtype=torch.long)
            eps = self._guided_eps(model, x, t, labels, guidance_scale)

            alpha_bar_t = _extract(self.alphas_cumprod, t, x.shape)
            x0_pred = (x - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
            x0_pred = x0_pred.clamp(-1, 1)

            if prev_t_int < 0:
                x = x0_pred
                continue

            prev_t = torch.full((shape[0],), prev_t_int, device=device, dtype=torch.long)
            alpha_bar_prev = _extract(self.alphas_cumprod, prev_t, x.shape)

            # Deterministic DDIM update (eta=0)
            x = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1.0 - alpha_bar_prev) * eps

        return x.clamp(-1, 1)
