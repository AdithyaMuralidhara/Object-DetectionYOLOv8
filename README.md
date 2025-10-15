# Real-Time Object Detection on Raspberry Pi 4B

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow Lite](https://img.shields.io/badge/TFLite-INT8-orange.svg)](https://www.tensorflow.org/lite)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Nano-green.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance object detection system for Raspberry Pi 4B that detects 4 common objects in real-time using a custom-trained YOLOv8n model optimized for embedded deployment.

##  Features

- **Real-time detection** at 25-30 FPS on Raspberry Pi 4B (CPU only)
- **Low latency**: 35-45ms inference time
- **4 object classes**: Mouse, Screwdriver, Sharpener, Pen
- **Optimized model** for speed
- **Reduced False Positive**
- **Color-coded boxes** for easy visualization
- **Google Colab Notebook** linked


##  Hardware Used

- **Raspberry Pi 4B** (4GB RAM recommended)
- **Pi Camera V1.3** (OV5647 sensor) or compatible
- **MicroSD card** (16GB minimum, 32GB recommended)
- **Power supply** (5V 3A USB-C)
- **Optional**: Heatsink/fan for sustained operation

##  Software Used

### Raspberry Pi
- **OS**: Ubuntu 22.04 LTS (64-bit)
- **Python**: 3.10+
- **Libraries**:
  - numpy 1.5
  - opencv-python 4.5
  - tflite-runtime
- **Camera**: rpicam-apps (libcamera)

### Training (Google Colab)
- **Python**: 3.10+
- **GPU**: T4 or better recommended
- **Libraries**:
  - ultralytics (YOLOv8)
  - torch, torchvision


