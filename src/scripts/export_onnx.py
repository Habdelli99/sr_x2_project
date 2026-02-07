import argparse
from pathlib import Path
import torch

from src.models.fsrcnn import FSRCNN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--size", type=int, default=128)  # LR input size
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = FSRCNN(scale=args.scale).to(device)
    state = torch.load(args.weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, args.size, args.size, device=device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["lr"],
        output_names=["sr"],
        dynamic_axes={"lr": {0: "batch", 2: "h", 3: "w"},
                      "sr": {0: "batch", 2: "H", 3: "W"}}
    )

    print("✅ Exported ONNX to:", out_path)

if __name__ == "__main__":
    main()
