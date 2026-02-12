#  AirLens-Vision: Real-Time Object Detection

**AirLens-Vision** is a high-speed computer vision application that identifies and labels objects in real-time. Built with the modern **MediaPipe Tasks API** and **OpenCV**, it utilizes the **EfficientDet-Lite0** model to provide professional-grade object detection directly on a live camera feed.

<img width="798" height="637" alt="ObjectDetect" src="https://github.com/user-attachments/assets/4bdf846c-e29b-4497-825e-7d06669ee77f" />


##  Features

* **Real-Time Inference**: Optimized for live video streams with minimal latency.
* **80+ Categories**: Detects a wide range of common objects (people, vehicles, laptops, bottles, etc.) based on the COCO dataset.
* **Dynamic UI**: Overlays precise bounding boxes, class labels, and confidence scores.
* **Lightweight Architecture**: Designed to run efficiently on standard hardware without requiring a dedicated high-end GPU.

##  Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/NKumar-B/ObjectSense-AI.git
cd ObjectSense-AI

```

### 2. Install Dependencies

Ensure you have Python 3.9+ installed, then run:

```bash
pip install -r requirements.txt

```

### 3. Download the Model

You must download the **EfficientDet-Lite0 (float32)** model and place it in the project root:

* **Model Name**: `efficientdet_lite0.tflite`
* **Source**: [Google MediaPipe Model Garden](https://www.google.com/search?q=https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector%23models).

##  How to Use

1. **Run the application**:
```bash
python ObjectDetect.py

```


2. **Interaction**:
* Point your webcam at objects to see bounding boxes and labels in real-time.
* The **Confidence Score** (0.0 - 1.0) indicates the AI's certainty.


3. **Exit**: Press **'q'** on your keyboard to close the window.

##  How It Works

1. **Preprocessing**: The input frame is mirrored and converted from BGR (OpenCV standard) to RGB (MediaPipe standard).
2. **Inference**: The frame is passed to the `ObjectDetector` task, which performs a single pass to identify multiple objects simultaneously.
3. **Visualization**: The result contains normalized coordinates which are mathematically mapped back to your screen's pixel dimensions to draw the bounding boxes accurately.

##  License

Distributed under the **MIT License**. See `LICENSE` for more information.

##  Acknowledgments

* Powered by [Google MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide).
* Trained on the [COCO Dataset](https://cocodataset.org/).

---
