fruit-detector/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── images/          # Your raw fruit images (train + val)
│   └── annotations/     # YOLO-format labels (.txt) for each image
│
├── configs/
│   └── fruit.yaml       # YOLO dataset config (paths + class names)
│
├── models/
│   └── best.pt          # Trained YOLOv8 model (created after training)
│
└── src/
    ├── train_fruit_detector.py
    ├── run_webcam.py
    └── utils.py


# 🍎🍌 Real-Time Fruit Detection (YOLOv8 + OpenCV)

A Python project that detects different kinds of fruit in **real time** from your webcam using **YOLOv8** and **OpenCV**.

## Features

- Real-time detection from webcam (apples, bananas, oranges, grapes, strawberries)
- Custom-trained YOLOv8 model on your own fruit dataset
- FPS counter overlay
- Clean, minimal code structure ready for GitHub

## Tech Stack

- Python 3.10+
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- OpenCV (`opencv-python`)
- PyTorch

## Project Structure

```text
fruit-detector/
├── README.md
├── requirements.txt
├── data/
│   ├── images/train
│   └── images/val
├── configs/
│   └── fruit.yaml
├── models/
│   └── best.pt
└── src/
    ├── train_fruit_detector.py



1. Setup
Clone and install
bash
git clone https://github.com/<your-username>/fruit-detector.git
cd fruit-detector

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
2. Prepare your dataset
Put training images in:

text
data/images/train/
Put validation images in:

text
data/images/val/
For each image, create a .txt file with the same name in YOLO format:

text
class_id x_center y_center width height
All coordinates normalized to [0, 1].

Edit configs/fruit.yaml if you change class names or paths.

3. Train the fruit detector
From the repo root:

bash
python src/train_fruit_detector.py \
  --data configs/fruit.yaml \
  --epochs 50 \
  --imgsz 640 \
  --model yolov8n.pt
Training logs and weights go to runs/fruit_train/exp/.

The script copies best.pt into:

text
models/best.pt
4. Run real-time detection from webcam
bash
python src/run_webcam.py
A window titled “Fruit Detector” will open.

Press q to quit.

5. Deploy & use
Local desktop (simple deployment)
Keep this repo on your machine.

Activate the virtual environment and run:

bash
python src/run_webcam.py
This is already a “deployment” for local use—great for demos and portfolio.

Optional: Docker (for portability)
Create a Dockerfile (optional):

dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "src/run_webcam.py"]
Build and run:

bash
docker build -t fruit-detector .
docker run --device /dev/video0:/dev/video0 -e DISPLAY=$DISPLAY fruit-detector
(Adjust camera device and display settings per OS.)
    ├── run_webcam.py
    └── utils.py
