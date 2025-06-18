from ultralytics import YOLO
import cv2

# Load the model
model = YOLO("v5.pt")  # Replace with your model path

# Input video
video_path = "bottle_demo4.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output video writer (1.5x speed = drop some frames or speed up)
output_path = "bottle_detection_1.5x.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps * 1.5, (width, height))

# Set normal window with half screen size
cv2.namedWindow("Bottle Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Bottle Detection", width // 2, height // 2)
cv2.moveWindow("Bottle Detection", 100, 100)  # Optional: position on screen

# Process video
frame_skip = 1  # You can increase to 2 for 2x speed
while cap.isOpened():
    for _ in range(frame_skip):
        cap.read()  # skip frames to speed up

    ret, frame = cap.read()
    if not ret:
        break

    # YOLO inference
    results = model(frame)
    annotated_frame = results[0].plot()

    # Write and show
    out.write(annotated_frame)
    cv2.imshow("Bottle Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
