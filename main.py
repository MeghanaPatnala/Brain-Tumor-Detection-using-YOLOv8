import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("best.pt path")

# Define image path
image_path = "brain.jpg path"

# Read the image
frame = cv2.imread(image_path)

# Check if the image is loaded correctly
if frame is None:
    print("Error: Could not load image. Check the file path.")
    exit()

# Confidence threshold for detection
CONFIDENCE_THRESHOLD = 0.5

# Run inference on the image
results = model.predict(frame, verbose=True)

# Ensure results contain detected objects
tumor_detected = False

if results and hasattr(results[0], "boxes") and results[0].boxes is not None:
    for box in results[0].boxes:
        confidence = box.conf[0].item()

        # Check confidence threshold
        if confidence > CONFIDENCE_THRESHOLD:
            tumor_detected = True
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())

            # Bounding box color (Bright Green) with increased thickness
            color = (0, 255, 0)  # Green
            thickness = 4  # Make bounding box more visible

            # Draw bounding box **only at tumor's location**
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)

            # Label text
            tumor_text = "TUMOR DETECTED"
            confidence_text = f"Confidence: {confidence:.2f}"

            # Draw label near the tumor
            cv2.putText(frame, tumor_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            cv2.putText(frame, confidence_text, (x_min, y_max + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

# If no tumor detected, display text on the image
if not tumor_detected:
    print("No Tumor Found.")
    cv2.putText(frame, "NO TUMOR FOUND", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
else:
    print("Tumor Found in the image.")

# Resize the image for display (optional)
frame_resized = cv2.resize(frame, (400, 400))

# Show the processed image
cv2.imshow("Brain Tumor Detection", frame_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
