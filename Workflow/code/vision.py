import cv2
from ultralytics import YOLO

# Load the TensorRT-optimized YOLO model
model = YOLO("yolo11n.engine")

# Open default webcam (0 is usually the built-in cam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("YOLOv8 TensorRT Live", frame)

    # Run YOLO inference
    results = model(rgb_frame)
    print(results)

    # Draw results
    for result in results:
        for box in result.boxes:
            #x1, y1, x2, y2 = map(int, box.xyxy[0])  # this was the problem
            x1 = int(box.xyxy[0][0].item())
            y1 = int(box.xyxy[0][1].item())
            x2 = int(box.xyxy[0][2].item())
            y2 = int(box.xyxy[0][3].item())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 30), 2)

    # Show the frame
    cv2.imshow("YOLOv8 TensorRT Live", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
