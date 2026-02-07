# SR ×2 — Real-Time Super-Resolution with FSRCNN (PyTorch / ONNX)

Super-resolution ×2 using FSRCNN with:
- Training on DIV2K
- PSNR / SSIM evaluation
- Real-time webcam demo
- ONNX export and FPS benchmark

---

## 📁 Project Structure

src/
├─ train.py # Training loop
├─ eval.py # PSNR / SSIM evaluation
├─ demo_live_split.py # Real-time demo (Original vs SR)
├─ models/fsrcnn.py # FSRCNN architecture
├─ datasets/ # DIV2K loader & preparation
├─ scripts/export_onnx.py # ONNX export
├─ scripts/test_onnx.py # ONNX inference benchmark
└─ configs/sr_x2_fsrcnn.yaml # Experiment configuration


---

## ⚙️ Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

🚀 Training
python -m src.train --config src/configs/sr_x2_fsrcnn.yaml

📊 Evaluation (PSNR / SSIM)
python -m src.eval --weights src/runs/<exp_name>/best.pt

🎥 Real-Time Demo

python -m src.demo_live_split --weights src/runs/<exp_name>/best.pt --size 192 --camera usb


⚡ ONNX Benchmark

python -m src.scripts.export_onnx --weights src/runs/<exp_name>/best.pt
python -m src.scripts.test_onnx

📝 Notes
data/, .venv/, runs/, model weights and outputs are intentionally excluded from Git.
Designed for GPU inference and Edge-AI performance testing.
---

### ✅ requirements.txt


orch
torchvision
opencv-python
numpy
pyyaml
onnx
onnxruntime


---

### ✅ Ensuite

```bash
git add README.md requirements.txt
git commit -m "Add clean README and requirements"
git push


