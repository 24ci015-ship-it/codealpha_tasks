import cv2
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

# Webcam
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

while True:
    success, frame = cap.read()

    if not success:
        break

    # Detection + Tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml"
    )

    annotated = results[0].plot()

    cv2.imshow("YOLOv8 Object Tracking", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()