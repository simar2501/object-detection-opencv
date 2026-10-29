# Real-Time Object Detection with OpenCV

Detects objects in webcam streams, video files, and images using a pre-trained MobileNet-SSD model running through OpenCV's DNN module. No GPU needed.

![Demo](outputs/demo.jpg)

## Features

- Three input modes: webcam (default), video file, single image
- Bounding boxes with class labels and confidence scores
- Live FPS and object count overlay
- Per-class confidence thresholds to cut false positives on classes that fire on clutter (plants, sofas, chairs)
- Optional saving of annotated video to `outputs/`

## Model

MobileNet-SSD (Caffe), trained on PASCAL VOC. It recognises 20 classes:
aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor.

Objects outside these classes (laptop, phone, cup, book) are not detected correctly. This is a limit of the model, not a bug. Model files are in `models/`. Original model by [chuanqi305/MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python detect.py                          # webcam
python detect.py --camera 1               # another camera
python detect.py --video path\to\clip.mp4 --save
python detect.py --image samples\test.jpg
```

Press `q` in the video window to quit. Image results are saved to `outputs/`.

## How it works

1. Each frame is resized to 300x300 and normalised into a blob.
2. The network outputs candidate boxes with class IDs and confidences.
3. Detections below the threshold are dropped (0.5 by default, stricter for noisy classes; see `CLASS_CONF` in `detect.py`).
4. Boxes are scaled back to the original frame size and drawn.

## Performance

Runs on CPU. FPS: XX on <your CPU>.

## Limitations

- Only 20 classes
- Small or distant objects are often missed (input is 300x300)
- No object tracking: each frame is detected independently

## Project structure

```
detect.py          detection script
models/            prototxt and caffemodel
samples/           test images
outputs/           saved results
requirements.txt
```