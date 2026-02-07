import argparse
import time
import cv2
import numpy as np
import torch

from src.models.fsrcnn import FSRCNN


def gstreamer_pipeline(width=1280, height=720, fps=30):
    return (
        "nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
    )


def center_crop(img, crop_size):
    h, w = img.shape[:2]
    cs = min(crop_size, h, w)
    x0 = (w - cs) // 2
    y0 = (h - cs) // 2
    return img[y0:y0+cs, x0:x0+cs]


def bgr_to_tensor_01(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    return x.unsqueeze(0)


def tensor_to_bgr_uint8(x):
    if x.dim() == 4:
        x = x[0]
    x = x.detach().clamp(0, 1).cpu().numpy()
    x = (x * 255.0).round().astype(np.uint8)
    x = np.transpose(x, (1, 2, 0))
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def put_fps(img, fps):
    text = f"FPS: {fps:.1f}"
    cv2.putText(img, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 0), 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--camera", type=str, default="usb",
                        choices=["csi", "usb"])
    parser.add_argument("--cam_w", type=int, default=1280)
    parser.add_argument("--cam_h", type=int, default=720)
    parser.add_argument("--cam_fps", type=int, default=30)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    use_fp16 = (device == "cuda")
    print("FP16:", use_fp16)


    # Load model
    model = FSRCNN(scale=2).to(device)
    state = torch.load(args.weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    if use_fp16:
       model.half()


    # Open camera
    if args.camera == "csi":
        cap = cv2.VideoCapture(
            gstreamer_pipeline(args.cam_w, args.cam_h, args.cam_fps),
            cv2.CAP_GSTREAMER
        )
        if not cap.isOpened():
            print("CSI not available -> fallback USB")
            cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    lr_size = args.size
    hr_size = lr_size * 2

    t_prev = time.time()
    fps = 0.0
    alpha = 0.1

    print("Press 'q' to quit.")

    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Prepare images
            orig_hr = center_crop(frame, hr_size)
            lr = cv2.resize(orig_hr, (lr_size, lr_size),
                            interpolation=cv2.INTER_CUBIC)

            #x = bgr_to_tensor_01(lr).to(device)
            #sr = model(x)

            x = bgr_to_tensor_01(lr).to(device)
            if use_fp16:
                x = x.half()
            sr = model(x)

            sr_bgr = tensor_to_bgr_uint8(sr)

            # --- FORCE SAME SIZE/TYPE ---
            lr_up = cv2.resize(lr, (hr_size, hr_size), interpolation=cv2.INTER_CUBIC)  # input visible
            left = lr_up
            right = cv2.resize(sr_bgr, (hr_size, hr_size))


            left = left.astype(np.uint8)
            right = right.astype(np.uint8)

            split = cv2.hconcat([left, right])

            # FPS
            t_now = time.time()
            inst_fps = 1.0 / max(1e-6, (t_now - t_prev))
            t_prev = t_now
            fps = (1 - alpha) * fps + alpha * inst_fps
            put_fps(split, fps)

            #cv2.imshow("SR x2 | Left: Original | Right: SR", split)
            cv2.imshow("SR x2 | Left: LR_up | Right: SR", split)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
