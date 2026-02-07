import torch
import torch.nn.functional as F
import math

def psnr(sr: torch.Tensor, hr: torch.Tensor, eps=1e-10) -> float:
    mse = F.mse_loss(sr, hr, reduction="mean").item()
    if mse < eps:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)

def _gaussian_kernel(window_size=11, sigma=1.5, device="cpu", dtype=torch.float32):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    k1 = g.view(1, 1, -1)
    k2 = k1.transpose(2, 1) @ k1  # (1,1,ws,ws)
    return k2

def ssim(sr: torch.Tensor, hr: torch.Tensor, window_size=11, sigma=1.5, C1=0.01**2, C2=0.03**2) -> float:
    device = sr.device
    dtype = sr.dtype
    kernel = _gaussian_kernel(window_size, sigma, device=device, dtype=dtype)

    def conv(x):
        B, C, H, W = x.shape
        k = kernel.expand(C, 1, window_size, window_size)
        return F.conv2d(x, k, padding=window_size//2, groups=C)

    mu_x = conv(sr)
    mu_y = conv(hr)
    sigma_x = conv(sr * sr) - mu_x * mu_x
    sigma_y = conv(hr * hr) - mu_y * mu_y
    sigma_xy = conv(sr * hr) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
    return ssim_map.mean().item()
