import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

SCALE = 2

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def center_crop_to_multiple(img: Image.Image, mult: int) -> Image.Image:
    w, h = img.size
    w2 = (w // mult) * mult
    h2 = (h // mult) * mult
    left = (w - w2) // 2
    top = (h - h2) // 2
    return img.crop((left, top, left + w2, top + h2))

def make_lr(hr: Image.Image, scale: int) -> Image.Image:
    w, h = hr.size
    lr_w, lr_h = w // scale, h // scale
    return hr.resize((lr_w, lr_h), resample=Image.BICUBIC)

def process_split(hr_dir: Path, out_input: Path, out_target: Path):
    ensure_dir(out_input)
    ensure_dir(out_target)

    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = [p for p in hr_dir.iterdir() if p.suffix.lower() in exts]
    files.sort()

    if not files:
        raise RuntimeError(f"Aucune image trouvée dans {hr_dir}")

    for p in tqdm(files, desc=f"Processing {hr_dir.name}"):
        hr = Image.open(p).convert("RGB")
        hr = center_crop_to_multiple(hr, SCALE)

        lr = make_lr(hr, SCALE)

        # Garder le même nom
        lr_path = out_input / p.name
        hr_path = out_target / p.name

        lr.save(lr_path, format="PNG")
        hr.save(hr_path, format="PNG")

def main():
    root = Path(__file__).resolve().parents[2]  # .../sr_x2_project
    raw = root / "data" / "div2k_raw"
    out = root / "data" / "div2k_x2"

    train_hr = raw / "DIV2K_train_HR"
    val_hr   = raw / "DIV2K_valid_HR"

    # Tu as déjà créé train/val/test. On va :
    # - train = DIV2K_train_HR
    # - val   = 80 images de valid
    # - test  = 20 images de valid (séparation simple)

    if not train_hr.exists():
        raise RuntimeError(f"Manque dossier: {train_hr}")
    if not val_hr.exists():
        raise RuntimeError(f"Manque dossier: {val_hr}")

    # Train complet
    process_split(
        train_hr,
        out / "train" / "input",
        out / "train" / "target"
    )

    # Split val/test depuis valid
    val_files = sorted([p for p in val_hr.iterdir() if p.suffix.lower() == ".png"])
    if len(val_files) < 20:
        raise RuntimeError("DIV2K_valid_HR doit contenir ~100 images PNG.")

    val_list = val_files[:80]
    test_list = val_files[80:]

    def process_list(file_list, out_input, out_target, name):
        ensure_dir(out_input)
        ensure_dir(out_target)
        for p in tqdm(file_list, desc=f"Processing {name}"):
            hr = Image.open(p).convert("RGB")
            hr = center_crop_to_multiple(hr, SCALE)
            lr = make_lr(hr, SCALE)
            lr.save(out_input / p.name, format="PNG")
            hr.save(out_target / p.name, format="PNG")

    process_list(val_list, out / "val" / "input", out / "val" / "target", "val")
    process_list(test_list, out / "test" / "input", out / "test" / "target", "test")

    print("\n✅ Dataset SR x2 prêt dans:", out)

if __name__ == "__main__":
    main()
