import argparse
import yaml
from pathlib import Path
import json
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from src.datasets.div2k_sr_loader import make_loaders
from src.models.fsrcnn import FSRCNN
from src.train_test.metrics import psnr, ssim

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.L1Loss()
    total = 0.0
    for lr, hr, _ in tqdm(loader, desc="train", leave=False):
        lr, hr = lr.to(device), hr.to(device)
        sr = model(lr)
        loss = loss_fn(sr, hr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    psnr_vals, ssim_vals = [], []
    for lr, hr, _ in tqdm(loader, desc="val", leave=False):
        lr, hr = lr.to(device), hr.to(device)
        sr = model(lr)
        psnr_vals.append(psnr(sr, hr))
        ssim_vals.append(ssim(sr, hr))
    return sum(psnr_vals)/len(psnr_vals), sum(ssim_vals)/len(ssim_vals)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    exp_dir = Path("src/runs") / cfg["exp_name"]
    exp_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_loader, val_loader = make_loaders(
        cfg["data_root"],
        train_bs=cfg["batch_size"],
        val_bs=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        hr_crop=cfg["train_size"],
        scale=cfg["scale"]
    )

    model = FSRCNN(scale=cfg["scale"]).to(device)
    optimizer = Adam(model.parameters(), lr=cfg["lr"])

    best_ssim = -1
    history = {"loss": [], "psnr": [], "ssim": []}

    for epoch in range(1, cfg["epochs"] + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        val_psnr, val_ssim = validate(model, val_loader, device)

        print(f"Epoch {epoch}: loss={loss:.4f} PSNR={val_psnr:.2f} SSIM={val_ssim:.4f}")

        history["loss"].append(loss)
        history["psnr"].append(val_psnr)
        history["ssim"].append(val_ssim)

        torch.save(model.state_dict(), exp_dir / "last.pt")
        if val_ssim > best_ssim:
            best_ssim = val_ssim
            torch.save(model.state_dict(), exp_dir / "best.pt")

    with open(exp_dir / "train_log.json", "w") as f:
        json.dump(history, f, indent=2)

    print("Training finished. Best model saved to:", exp_dir / "best.pt")

if __name__ == "__main__":
    main()
