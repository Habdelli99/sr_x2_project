from src.datasets.div2k_sr_loader import make_loaders

def main():
    train_loader, val_loader = make_loaders(
        "data/div2k_x2",
        train_bs=2,
        val_bs=2,
        num_workers=0,   # IMPORTANT sur Windows pour éviter spawn issues
        hr_crop=256,
        scale=2
    )

    lr, hr, names = next(iter(train_loader))
    print("LR:", lr.shape, float(lr.min()), float(lr.max()))
    print("HR:", hr.shape, float(hr.min()), float(hr.max()))
    print("Names:", names[:2])

if __name__ == "__main__":
    main()
