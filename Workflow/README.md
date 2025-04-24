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

 While the program was working initially, due to unknown reasons, it no longer produced a video feed. However, we were able to gather some results indicating that it was using GPU acceleration:
![2dea4f87-d75c-495c-96e5-9c835da27b14](https://github.com/user-attachments/assets/a815103b-a0f5-4247-bd57-584f127b8f68)

Additionally, we were able to capture a video from another device of the initial run.
Here is a screenshot from the video showing the GPU utilization, as well as a frame of the results:
![image](https://github.com/user-attachments/assets/ff25727c-a6e1-4435-83e6-1ea3179fcb58)





