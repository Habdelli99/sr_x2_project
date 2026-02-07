import argparse
import yaml
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np

from src.datasets.div2k_sr_loader import Div2KSRPaired
from src.models.fsrcnn import FSRCNN
from src.train_test.metrics import psnr, ssim


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def tensor_to_uint8_img(x):
    x = x.detach().cpu().clamp(0, 1).numpy()
    x = (x * 255.0).round().astype(np.uint8)
    x = np.transpose(x, (1, 2, 0))  # HWC RGB
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    return x


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--num_samples", type=int, default=16)
    args = parser.parse_args()

    cfg = load_config(args.config)

    exp_dir = Path("src/runs") / cfg["exp_name"]
    samples_dir = exp_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ds = Div2KSRPaired(
        cfg["data_root"],
        args.split,
        scale=cfg["scale"],
        hr_crop=cfg["train_size"],
        augment=False,
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

    model = FSRCNN(scale=cfg["scale"]).to(device)
    state = torch.load(args.weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    psnr_list, ssim_list = [], []
    saved = 0

    for lr, hr, names in loader:
        lr, hr = lr.to(device), hr.to(device)
        sr = model(lr)

        psnr_list.append(psnr(sr, hr))
        ssim_list.append(ssim(sr, hr))

        for i in range(lr.shape[0]):
            if saved >= args.num_samples:
                break

            lr_img = tensor_to_uint8_img(lr[i])
            hr_img = tensor_to_uint8_img(hr[i])
            sr_img = tensor_to_uint8_img(sr[i])

            h, w = hr_img.shape[:2]
            lr_up = cv2.resize(lr_img, (w, h), interpolation=cv2.INTER_CUBIC)

            grid = cv2.hconcat([lr_up, sr_img, hr_img])
            out_path = samples_dir / f"{saved:02d}_{names[i].replace('.png','')}.png"
            cv2.imwrite(str(out_path), grid)
            saved += 1

    mean_psnr = float(sum(psnr_list) / len(psnr_list))
    mean_ssim = float(sum(ssim_list) / len(ssim_list))

    print(f"{args.split} metrics: PSNR={mean_psnr:.2f} SSIM={mean_ssim:.4f}")
    print("Saved samples to:", samples_dir)

    metrics_path = exp_dir / "eval_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {"split": args.split, "psnr": mean_psnr, "ssim": mean_ssim},
            f,
            indent=2,
        )
    print("Saved metrics to:", metrics_path)


if __name__ == "__main__":
    main()
