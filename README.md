
# Neural Network Object Detection with OpenCV DNN

This project uses **pretrained TensorFlow models** with OpenCV’s `dnn` module to detect **faces** and **people** in a dataset of images. It supports automated classification, result visualization, and evaluation using confusion matrices.

> Developed as part of CSC-FPX4040 at Capella University  
> Demonstrates integration of neural network inference into OpenCV pipelines

---

## Features

- Face and person detection using pretrained `.pb` models
- Uses `cv2.dnn.readNetFromTensorflow` for inference
- Automatically classifies images into `positive/` and `negative/`
- Evaluates predictions with manifest-based ground truth
- Visualizes confusion matrices for performance insights

---

## Project Structure

```bash
.
├── Assessment6_NeuralNetworkDetection.ipynb   # Main notebook
├── resources-nn_models/
│   ├── opencv_face_detector_uint8.pb
│   ├── opencv_face_detector.pbtxt
│   ├── frozen_inference_graph.pb
│   ├── ssd_inception_v2_coco_2017_11_17.pbtxt
│   └── classes.json
├── resources-nn_dataset_1/
│   ├── *.jpg / *.png                         # Test images
│   ├── nn_dataset_1_face_manifest.txt
│   └── nn_dataset_1_person_manifest.txt
├── output/
│   ├── faces/
│   │   ├── positive/
│   │   └── negative/
│   ├── persons/
│   │   ├── positive/
│   │   └── negative/
│   ├── face_confusion_matrix.png
│   └── person_confusion_matrix.png
├── requirements.txt
└── README.md
```

---

## Setup Instructions

1. **Install requirements:**

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install opencv-python pandas matplotlib scikit-learn
```

2. **Run the notebook:**

```bash
jupyter notebook Assessment6_NeuralNetworkDetection.ipynb
```

You’ll need:
- `.pb` and `.pbtxt` model files in `resources-nn_models/`
- A folder of images to analyze
- Manifest `.txt` files listing true positives

---

## How It Works

### Face Detection
- Model: `opencv_face_detector_uint8.pb`
- Input processed with `cv2.dnn.blobFromImage`
- Bounding boxes drawn for confidence > 0.5

### Person Detection
- Model: `SSD Inception v2 COCO`
- Class labels loaded from `classes.json`
- Only “person” class detected and visualized

---

## Output Example

| Task              | Output |
|-------------------|--------|
| Face Detection    | `output/faces/positive/*.jpg` with green boxes |
| Person Detection  | `output/persons/positive/*.jpg` with blue boxes |
| Performance Eval  | `face_confusion_matrix.png` and `person_confusion_matrix.png` |

---

## Learning Goals

- Apply neural networks to real-world image classification
- Learn model loading and inference with OpenCV’s DNN API
- Evaluate accuracy with true/false positive breakdowns
- Understand data preprocessing for neural networks

---

## Acknowledgements

- TensorFlow Object Detection Zoo  
- OpenCV’s pretrained DNN face model  
- Capella University CSC-FPX4040 coursework

---

## License

This project is for academic and educational purposes.  
Reuse and adaptation encouraged with attribution.

---
