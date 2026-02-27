"""
FID (Fréchet Inception Distance) Evaluation
Lightweight implementation using torchvision InceptionV3.
Works on MNIST/Fashion-MNIST/CIFAR-10.

Usage:
    from fid import FIDEvaluator
    fid_eval = FIDEvaluator(device=device, dataset="fashion_mnist")
    fid_eval.compute_real_stats(real_loader)        # once per dataset
    score = fid_eval.compute_fid(model, diffusion, tcfg, n_samples=5000)
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import linalg
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# ─────────────────────────────────────────────
# InceptionV3 Feature Extractor
# ─────────────────────────────────────────────
class InceptionFeatureExtractor(nn.Module):
    """
    Extract 2048-dim pool3 features from InceptionV3.
    Handles grayscale (MNIST) by repeating channels.
    Input: (B, C, H, W) in [-1, 1] or [0, 1]
    Output: (B, 2048)
    """
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        # Remove final classifier, keep up to pool3
        self.blocks = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        # x: (B, C, H, W), values in [-1,1]
        # Inception expects [0,1], 3-channel, >= 75px
        x = (x * 0.5 + 0.5).clamp(0, 1)          # [-1,1] → [0,1]
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)              # grayscale → RGB
        x = nn.functional.interpolate(x, size=(299, 299),
                                       mode="bilinear", align_corners=False)
        features = self.blocks(x)
        return features.flatten(1)                 # (B, 2048)


# ─────────────────────────────────────────────
# FID Math
# ─────────────────────────────────────────────
def compute_statistics(features: np.ndarray):
    """Mean and covariance of feature matrix (N, D)."""
    mu  = features.mean(axis=0)
    sig = np.cov(features, rowvar=False)
    return mu, sig


def frechet_distance(mu1, sig1, mu2, sig2, eps=1e-6):
    """FID between two Gaussians N(mu1, sig1) and N(mu2, sig2)."""
    diff = mu1 - mu2
    # Stable sqrt of product of covariances
    covmean, _ = linalg.sqrtm(sig1 @ sig2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sig1.shape[0]) * eps
        covmean = linalg.sqrtm((sig1 + offset) @ (sig2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError("Imaginary component in sqrtm result")
        covmean = covmean.real
    return float(diff @ diff + np.trace(sig1) + np.trace(sig2) - 2 * np.trace(covmean))


# ─────────────────────────────────────────────
# FID Evaluator
# ─────────────────────────────────────────────
class FIDEvaluator:
    def __init__(self, device, batch_size=128):
        self.device     = device
        self.batch_size = batch_size
        self.extractor  = InceptionFeatureExtractor().to(device).eval()
        self.real_mu    = None
        self.real_sig   = None

    @torch.no_grad()
    def extract_features(self, loader_or_tensor):
        """
        loader_or_tensor: DataLoader yielding (images, labels)
                          OR Tensor of shape (N, C, H, W) in [-1,1]
        Returns: np.ndarray (N, 2048)
        """
        feats = []
        if isinstance(loader_or_tensor, torch.Tensor):
            ds = TensorDataset(loader_or_tensor)
            loader = DataLoader(ds, batch_size=self.batch_size)
            for (x,) in tqdm(loader, desc="  FID features (gen)", leave=False):
                feats.append(self.extractor(x.to(self.device)).cpu().numpy())
        else:
            for x, _ in tqdm(loader_or_tensor, desc="  FID features (real)", leave=False):
                feats.append(self.extractor(x.to(self.device)).cpu().numpy())
        return np.concatenate(feats, axis=0)

    def compute_real_stats(self, real_loader):
        """Compute and cache real data statistics (call once per dataset)."""
        print("Computing real data statistics for FID...")
        feats = self.extract_features(real_loader)
        self.real_mu, self.real_sig = compute_statistics(feats)
        print(f"  Real stats computed from {len(feats)} samples.")

    @torch.no_grad()
    def compute_fid(self, model, diffusion, tcfg,
                    n_samples=5000, sampler="ddim", steps=50,
                    guidance_scale=3.0, seed=42):
        """
        Generate n_samples images and compute FID against cached real stats.
        Returns: float FID score
        """
        assert self.real_mu is not None, "Call compute_real_stats() first."
        model.eval()

        num_classes  = tcfg.get("num_classes", 10)
        img_size     = tcfg.get("img_size", 28)
        in_channels  = tcfg.get("in_channels", 1)
        batch_size   = self.batch_size
        torch.manual_seed(seed)

        all_samples = []
        n_generated = 0
        print(f"Generating {n_samples} samples for FID ({sampler}, steps={steps})...")

        while n_generated < n_samples:
            bs = min(batch_size, n_samples - n_generated)
            # Cycle through classes evenly
            labels = torch.tensor(
                [i % num_classes for i in range(n_generated, n_generated + bs)],
                device=self.device, dtype=torch.long
            )
            shape = (bs, in_channels, img_size, img_size)

            if sampler == "ddpm":
                samples = diffusion.ddpm_sample(model, shape, labels,
                                                 guidance_scale=guidance_scale)
            else:
                samples = diffusion.ddim_sample(model, shape, labels,
                                                 steps=steps,
                                                 guidance_scale=guidance_scale)
            all_samples.append(samples.cpu())
            n_generated += bs

        gen_tensor = torch.cat(all_samples, dim=0)[:n_samples]
        gen_feats  = self.extract_features(gen_tensor)
        gen_mu, gen_sig = compute_statistics(gen_feats)

        fid = frechet_distance(self.real_mu, self.real_sig, gen_mu, gen_sig)
        return fid


if __name__ == "__main__":
    # Quick smoke test (no GPU needed)
    import torch
    device = torch.device("cpu")
    extractor = InceptionFeatureExtractor().to(device).eval()
    x = torch.randn(4, 1, 28, 28)
    with torch.no_grad():
        f = extractor(x)
    print("Feature shape:", f.shape)   # (4, 2048)
    print("FID module OK")
