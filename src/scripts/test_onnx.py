import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("src/runs/div2k_x2_fsrcnn_baseline/model.onnx")

lr = np.random.rand(1,3,128,128).astype(np.float32)
sr = sess.run(None, {"lr": lr})[0]

print("ONNX output shape:", sr.shape)
