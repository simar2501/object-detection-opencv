import argparse
import time
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).parent
PROTOTXT = BASE / "models" / "deploy.prototxt"
WEIGHTS = BASE / "models" / "mobilenet_iter_73000.caffemodel"
OUT_DIR = BASE / "outputs"

# The 20 PASCAL VOC classes this model was trained on (index 0 is background)
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

DEFAULT_CONF = 0.5
# Classes that tend to fire on random clutter get a stricter cutoff.
# These are starting values; adjust them after testing on your own footage.
CLASS_CONF = {
    "pottedplant": 0.65,
    "sofa": 0.65,
    "chair": 0.60,
    "diningtable": 0.60,
    "tvmonitor": 0.60,
    "boat": 0.60,
    "bottle": 0.55,
}

COLORS = np.random.default_rng(7).integers(60, 200, size=(len(CLASSES), 3)).tolist()


def load_net():
    if not PROTOTXT.exists() or not WEIGHTS.exists():
        raise SystemExit("Model files are missing from models/. See the README.")
    return cv2.dnn.readNetFromCaffe(str(PROTOTXT), str(WEIGHTS))


def detect(net, frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    out = net.forward()

    found = []
    for i in range(out.shape[2]):
        class_id = int(out[0, 0, i, 1])
        conf = float(out[0, 0, i, 2])
        if class_id <= 0 or class_id >= len(CLASSES):
            continue
        name = CLASSES[class_id]
        if conf < CLASS_CONF.get(name, DEFAULT_CONF):
            continue
        x1, y1, x2, y2 = (out[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
        found.append((name, conf, max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)))
    return found


def draw(frame, found):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for name, conf, x1, y1, x2, y2 in found:
        color = COLORS[CLASSES.index(name)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {conf * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, font, 0.55, 1)
        top = max(y1, th + 8)
        cv2.rectangle(frame, (x1, top - th - 8), (x1 + tw + 6, top), color, -1)
        cv2.putText(frame, label, (x1 + 3, top - 5), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def run_image(net, path):
    frame = cv2.imread(str(path))
    if frame is None:
        raise SystemExit(f"Could not read image: {path}")

    found = detect(net, frame)
    draw(frame, found)

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"{Path(path).stem}_detected.jpg"
    cv2.imwrite(str(out_path), frame)

    print(f"{len(found)} object(s) found:")
    for name, conf, *_ in found:
        print(f"  {name}: {conf * 100:.1f}%")
    print("Saved:", out_path)

    cv2.imshow("Detections (press any key to close)", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_stream(net, source, save):
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise SystemExit("Could not open the video source.")

    writer = None
    fps = 0.0
    prev = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        found = detect(net, frame)
        draw(frame, found)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
        prev = now
        cv2.rectangle(frame, (0, 0), (230, 28), (30, 30, 30), -1)
        cv2.putText(
            frame, f"{fps:.0f} FPS | {len(found)} objects", (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

        if save:
            if writer is None:
                OUT_DIR.mkdir(exist_ok=True)
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(OUT_DIR / "output.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h)
                )
            writer.write(frame)

        cv2.imshow("Object detection (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Object detection with OpenCV and MobileNet-SSD")
    parser.add_argument("--image", help="path to an image")
    parser.add_argument("--video", help="path to a video file")
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    parser.add_argument("--save", action="store_true", help="save the video output to outputs/")
    args = parser.parse_args()

    net = load_net()

    if args.image:
        run_image(net, args.image)
    elif args.video:
        run_stream(net, args.video, args.save)
    else:
        run_stream(net, args.camera, args.save)


if __name__ == "__main__":
    main()