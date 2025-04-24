# Accelerating Real-Time Object Detection on Embedded GPU Devices

## Problem Statement

Real-time object detection is a key AI task in applications such as autonomous vehicles, surveillance, robotics, and industrial automation. These tasks are computationally intensive and have traditionally relied on cloud-based servers. However, there is a growing demand for **low-latency and on-device AI inference**.

Edge AI platforms with GPU acceleration offer a solution: enabling AI tasks to be performed closer to the data source with higher speed, lower latency, and better efficiency. The goal of this project is to **leverage GPU-accelerated embedded hardware (NVIDIA Jetson Orin Nano)** to prototype a pipeline for **real-time object detection** using a pre-trained deep learning model.

---

## Initial Configuration and Setup

Our setup involves the following:

### Hardware:
- **NVIDIA Jetson Orin Nano Developer Kit**
- **USB Camera**
- **256GB microSD card** 
- Power supply, monitor, keyboard/mouse

### Software:
- **JetPack 6.1** (includes L4T, CUDA, cuDNN, TensorRT, and OpenCV)
- **Ultralytics YOLOv8**
- **Python**
- **PyTorch with Jetson GPU support**
- **OpenCV**
- Git and VSCode for code editing

---

## Libraries, Frameworks, and Tools

Here are the tools considered and used:

| Category        | Options Researched                            | Final Selection         | Reasoning                                |
|----------------|-----------------------------------------------|-------------------------|------------------------------------------|
| Framework      | TensorRT, ONNX Runtime, OpenVINO              | **TensorRT**            | Native to Jetson, GPU-accelerated        |
| Model      | YOLOv8, MobileNet-SSD            | **YOLOv8n**             | Optimized, fast, highly accurate         |
| Preprocessing  | OpenCV, TorchVision                      | **OpenCV**       | Easy integration, real-time performance  |
| Profiling      | `tegrastats`, `jtop`, `perf`, `nvprof`        | **jtop**          | Light and Jetson-native resource monitor |

---

## Pre-trained Model Choice

We selected **YOLOv8n** (nano version) from the [Ultralytics YOLO repository](https://github.com/ultralytics/ultralytics). Reasons:
- Lightweight, optimized for real-time edge applications
- Strong accuracy-performance balance
- Easy to export to TensorRT
- Well-supported in the Ultralytics Python API
