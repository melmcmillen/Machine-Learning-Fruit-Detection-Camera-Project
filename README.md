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
