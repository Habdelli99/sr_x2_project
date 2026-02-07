@"
# sr_x2_project — Super-Resolution x2 (FSRCNN)

Real-time and offline super-resolution (×2) using FSRCNN.  
Includes training, evaluation (PSNR/SSIM), live webcam demo, and optional ONNX export + benchmark.

## Project structure
- `src/train.py` — training loop
- `src/eval.py` — evaluation (PSNR/SSIM)
- `src/demo_live_split.py` — real-time demo (original vs SR)
- `src/models/fsrcnn.py` — FSRCNN model
- `src/datasets/` — DIV2K loader + prep scripts
- `src/scripts/export_onnx.py` — export to ONNX
- `src/scripts/test_onnx.py` — ONNX inference benchmark
- `src/configs/sr_x2_fsrcnn.yaml` — experiment config

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
