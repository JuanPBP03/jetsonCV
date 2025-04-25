# Edge AI Object Detection on the Jetson
## Step 0: Flash Jetpack onto Jetson

Following [Nvidia's guide](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write) we flashed an SD card with Jetpack 6.1.
### Step 0.1
Setup github on Jetson for an organized workflow.
```bash
sudo apt install gh
gh auth login
gh repo create
```

### Step 0.2: Choose a pre-trained AI model.
We decided to use YOLO11 since Ultralytics has a guide to [setup YOLO on the Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/). We chose to install packages natively as opposed to using a Docker container.

*Folowing steps are derived from the guide*
## Step 1: Update packages and install Ultralytics

### 1.1 The first step was to update the packages on the Jetson
```bash
sudo apt update
sudo apt install python3-pip -y
pip install -U pip
```
### 1.2 Next we installed Ultralytics with the necessary dependencies
```bash
pip install ultralytics[export]
```
### 1.3 Reboot the Jetson 
```bash
sudo reboot
```
## Step 2: Install PyTorch and Torchvision

While the previous step installed Torch and Torchvision, they are not compatible with Jetson due to its ARM64 architecture. Ulralytics therefore has us manually install their versions of PyTorch and Torchvision for Jetpack 6.1.
### 2.1 Uninstall current versions
```bash
pip uninstall torch torchvision
```
### 2.2 Install cuSPARSELt
To fix a dependency issue with torch 2.5.0, we install cuSPARSELt, an NVIDIA made Cuda library for sparse matrix multiplication.
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install libcusparselt0 libcusparselt-dev
```
## Step 3: Install onnxruntime-gpu
onnxruntime is a scoring engine for Open Neural Network Exchange (ONNX). Specifically, we are using the gpu version which takes advantage of the GPU's hardware acceleration to achieve faster inferencing.
```bash
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl
```

## Step 4: Convert model to TensorRT

Due to TensorRT's high performance on Jetson devices, we exported the YOLO11 model from PyTorch format to TensorRT.
```bash
# Executed in repo directory
yolo export model=yolo11n.pt format=engine # creates 'yolo11n.engine'
```

## Step 5: Develop Python script

Using OpenCV we developped a python script to perform inference with our model frame-by-frame. It follows the typical OpenCV workflow of opening the video feed, reading each frame, extracting data from the frame, and using that data to draw on the original frame of the video. In this case, our video feed is from the webcam provided, and we extract data by passing each frame to the model with ```results = model(frame)```. Each results object contains the bounding boxes of objects detected in the frame, and each box object contains the location information in various formats, the confidence of the detection, and metadata such as the id corresponding to the label it has detected (for more information see [here](https://docs.ultralytics.com/modes/predict/#boxes)). Using this information we used OpenCV functions including ```rectangle()``` and ```putText``` to draw the boxes, labels, and confidence values on each frame.

## Results
*see video in this folder for full video*
While the program was working initially, due to unknown reasons, it no longer produced a video feed. However, we were able to gather some results indicating that it was using GPU acceleration:
![2dea4f87-d75c-495c-96e5-9c835da27b14](https://github.com/user-attachments/assets/a815103b-a0f5-4247-bd57-584f127b8f68)

Additionally, we were able to capture a video from another device of the initial run.
Here is a screenshot from the video showing the GPU utilization, as well as a frame of the results:
![image](https://github.com/user-attachments/assets/ff25727c-a6e1-4435-83e6-1ea3179fcb58)

## Fixing Errors
Although our original workflow worked the first time we ran it. It stopped working during subsequent trials. After some troubleshooting, it seemed like the problem was encountered when we were trying to get the bounding box coordinates using the map function.
```
x1, y1, x2, y2 = map(int, box.xyxy[0])
```
For some reason, the map function was halting the execution of the while loop.

If you run into this issue, then delete the line that uses the map function and get the bounding box coordinates directly from the tensor using the item() method.
```
x1 = int(box.xyxy[0][0].item())
y1 = int(box.xyxy[0][1].item())
x2 = int(box.xyxy[0][2].item())
y2 = int(box.xyxy[0][3].item())
```
# New Results
### Object Detection Result:
![image](https://github.com/user-attachments/assets/79780d78-f567-4532-aedc-7d8e33aff9f6)

### Inference Time:
![image](https://github.com/user-attachments/assets/c10e2514-f274-463b-80f7-addad1adf7e5)

### GPU utilization before running the object detection:
![Screenshot from 2025-04-25 15-00-34](https://github.com/user-attachments/assets/d7f7fc6d-44df-42f0-8ee1-2ab24dc069b3)

### GPU utilization after running the object detection:
![image](https://github.com/user-attachments/assets/45e0ff42-0518-40f5-b0c0-bc04531a5185)

