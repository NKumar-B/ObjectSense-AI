# AirLens - Object Detection
#  Libraries to be Installed 
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#  1. DETECTOR CONFIGURATION 
# Path to the pre-trained EfficientDet-Lite0 TFLite model.
# Ensure 'efficientdet_lite0.tflite' is in your project directory.
model_path = 'efficientdet_lite0.tflite' 
base_options = python.BaseOptions(model_asset_path=model_path)

# Configure the detector options:
#  score_threshold: Only show objects detected with >50% confidence.
#  running_mode: Set to IMAGE for synchronous processing of video frames.
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    score_threshold=0.5, 
    running_mode=vision.RunningMode.IMAGE
)
# Create the detector instance using the defined options.
detector = vision.ObjectDetector.create_from_options(options)

#  2. CAMERA SETUP 
# Initialize the webcam feed (0 is usually the default internal camera).
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        break

    # Mirror the frame horizontally for a more natural 'selfie' view.
    frame = cv2.flip(frame, 1)
    
    # MediaPipe requires RGB images, but OpenCV captures in BGR by default.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert the processed RGB frame into a MediaPipe Image object.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    #  3. RUN INFERENCE 
    # Detect objects within the current frame.
    detection_result = detector.detect(mp_image)

    #  4. VISUALIZE RESULTS
    # Iterate through every object detected in the current frame.
    for detection in detection_result.detections:
        # Extract Bounding Box details.
        bbox = detection.bounding_box
        start = int(bbox.origin_x), int(bbox.origin_y)
        end = int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height)
        
        # Draw a Blue rectangle around the detected object.
        cv2.rectangle(frame, start, end, (255, 0, 0), 2)

        # Retrieve the category name (e.g., 'person', 'cup') and confidence score.
        category = detection.categories[0]
        text = f"{category.category_name} ({round(category.score, 2)})"
        
        # Overlay the text label above the bounding box.
        cv2.putText(frame, text, (start[0], start[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the final frame with overlays.
    cv2.imshow("AirLens - Vision", frame)
    
    # Exit the loop if the 'q' key is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# Cleanup resources.
cap.release()

cv2.destroyAllWindows()
