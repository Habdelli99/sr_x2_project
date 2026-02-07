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

src/
├─ train.py
├─ eval.py
├─ demo_live_split.py
├─ models/fsrcnn.py
├─ datasets/
├─ scripts/export_onnx.py
└─ configs/sr_x2_fsrcnn.yaml

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision opencv-python pyyaml onnxruntime
Train
python -m src.train --config src/configs/sr_x2_fsrcnn.yaml
Evaluate
python -m src.eval --weights src/runs/<exp_name>/best.pt
Real-time demo
python -m src.demo_live_split --weights src/runs/<exp_name>/best.pt --size 192 --camera usb
ONNX benchmark
python -m src.scripts.export_onnx --weights src/runs/<exp_name>/best.pt
python -m src.scripts.test_onnx
data/, .venv/, runs/ and weights are excluded from git.


Enregistre.

---

## 2) Ajoute `requirements.txt`

```powershell
notepad requirements.txt
Colle :

torch
torchvision
opencv-python
numpy
pyyaml
onnx
onnxruntime
3) Commit & push
git add README.md requirements.txt
git commit -m "Add README and requirements"
git push
4) Résultat attendu sur GitHub
Recharge la page :
Le README s’affiche → ton repo devient immédiatement compréhensible.

5) Étape suivante (qui fait passer de “TP” à “Edge-AI project”)
Après ça, on ajoute :

une section Results (PSNR / SSIM / FPS 256 vs 192 vs 128)

2 screenshots du demo

un tableau perf PyTorch vs ONNX
