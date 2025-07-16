import cv2
import numpy as np
from ultralytics import YOLO
import os
import json
import torch

class TrafficLightDetector:
    def __init__(self, video_path, output_dir="output", model_name="yolov8n.pt", detection_interval=0.25):
        """Initialize the traffic light object detector with YOLOv8 and video input."""
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")
        self.image_size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                          int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.timestep = 1.0 / self.fps if self.fps > 0 else 1.0  # Frame interval in seconds

        # Device setup
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"TrafficLightDetector using device: {self.device}")

        # Load YOLOv8 model
        try:
            self.model = YOLO(model_name)
            self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLOv8 model: {e}")

        # Logging setup
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.objects_log_file = os.path.join(self.output_dir, "objects_log.json")
        self.objects_log = []

        # Detection interval setup (in seconds)
        self.detection_interval = detection_interval  # Time between detections
        self.last_detection_time = 0.0  # Track last detection time
        self.frame_count = 0  # Track frame number for timing

    def run(self):
        """Main loop to read video frames, detect objects occasionally, and display results."""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("End of video or failed to read frame")
                break

            # Convert frame to BGR (if needed, OpenCV reads in BGR by default)
            rgb_image = frame  # Already in BGR format from OpenCV

            # Calculate current time based on frame count
            current_time = self.frame_count * self.timestep
            self.frame_count += 1

            # Perform detection only if the interval has passed
            detected_objects = []
            if current_time - self.last_detection_time >= self.detection_interval:
                detected_objects = self.process_image(rgb_image)
                self.last_detection_time = current_time  # Update last detection time

            # Draw bounding boxes and labels on the image (only for new detections)
            annotated_image = self.draw_detections(rgb_image, detected_objects)

            # Display the annotated image
            cv2.imshow("Object Detection", annotated_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key press
                break

            # Save detected objects to log if any were detected
            if detected_objects:
                self.objects_log.extend(detected_objects)
                self.save_log()

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

    def process_image(self, rgb_image):
        """Process a single image with YOLOv8 and return detected objects."""
        if rgb_image is None or rgb_image.size == 0:
            print("Invalid RGB image: None or empty")
            return []

        # Run YOLOv8 inference
        try:
            results = self.model(rgb_image, device=self.device, verbose=False)
        except Exception as e:
            print(f"YOLOv8 inference failed: {e}")
            return []

        detected_objects = []
        timestamp = self.frame_count * self.timestep

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                print("No objects detected")
                continue

            for box in boxes:
                try:
                    class_id = int(box.cls)
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf)
                    bbox = box.xyxy.cpu().numpy().astype(int).flatten().tolist()  # [x1, y1, x2, y2]

                    # Validate bbox coordinates
                    if (bbox[0] < 0 or bbox[1] < 0 or 
                        bbox[2] > self.image_size[0] or bbox[3] > self.image_size[1] or
                        bbox[2] <= bbox[0] or bbox[3] <= bbox[1]):
                        print(f"Invalid bbox coordinates: {bbox}")
                        continue

                    # Create object data
                    obj_data = {
                        'timestamp': timestamp,
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': bbox
                    }
                    detected_objects.append(obj_data)
                except Exception as e:
                    print(f"Error processing box: {e}")
                    continue

        return detected_objects

    def draw_detections(self, image, detected_objects):
        """Draw bounding boxes and labels on the image."""
        annotated_image = image.copy()
        for obj in detected_objects:
            bbox = obj['bbox']
            class_name = obj['class_name']
            confidence = obj['confidence']
            # Draw bounding box
            cv2.rectangle(annotated_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            # Draw label
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(annotated_image, label, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return annotated_image

    def save_log(self):
        """Save the objects log to a JSON file."""
        try:
            with open(self.objects_log_file, 'w') as f:
                json.dump(self.objects_log, f, indent=2)
            print(f"Objects log saved to {self.objects_log_file}")
        except Exception as e:
            print(f"Failed to save objects log: {e}")

def main():
    """Initialize and run the traffic light detector."""
    video_path = r"C:\Users\HP\Desktop\DSAIL\acvss\vehicles\controllers\camera\real_traffic.mp4"  # Replace with your video file path
    detector = TrafficLightDetector(video_path=video_path, detection_interval=1.0)
    try:
        detector.run()
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

