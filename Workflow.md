# Edge AI Object Detection on the Jetson

## Step 1
Setup github on Jetson for an organized workflow.
```python
# Code block
```

## Step 2
Choose an pre-trained AI model.
We decided to choose MobileNetV1 due to finding a guide that used a MobileNet model on the Jetson previously, https://www.theseus.fi/bitstream/handle/10024/745901/Nummela_Tino.pdf?sequence=2

## Step 3
Ask ChatGPT for steps on how to implement object detection.
##### ChatGPT Prompt:
_write us python code to do object detection with bounding boxes using an MobileNetV1 AI model on the Jetson. Please make the code so that the gpu is used for the computations. The code needs to use the camera to do object detection_

## Step 4
Implement instructions from ChatGPT.
Fist, we are required to install the TensorFlow library.
```bash
pip install tensorflow-gpu opencv-python opencv-python-headless
```
While installing the library, we found that tensorflow-gpu is the same as tensorflow, and that tensorflow-gpu doesn't exist anymore.

## S


import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Set GPU to be used by TensorFlow
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load the object detection model (MobileNetV1)
MODEL_PATH = 'ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
LABEL_MAP_PATH = 'mscoco_label_map.pbtxt'

# Load the pre-trained model
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(MODEL_PATH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Load label map
category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH, use_display_name=True)

# Start video capture from the camera (use 0 for the default camera)
cap = cv2.VideoCapture(0)

# Start TensorFlow session
with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        # Get tensor references for input and output
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        while True:
            # Read the frame from the camera
            ret, frame = cap.read()
            if not ret:
                break

            # Expand dimensions to match the input size of the model
            image_np = np.expand_dims(frame, axis=0)

            # Run inference
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np})

            # Visualize the results (bounding boxes on the image)
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                instance_masks=None,
                use_normalized_coordinates=True,
                line_thickness=8)

            # Display the frame with bounding boxes
            cv2.imshow('Object Detection', frame)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the capture and close the window
        cap.release()
        cv2.destroyAllWindows()
