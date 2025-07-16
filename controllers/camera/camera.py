import cv2
import numpy as np
from controller import Robot
from ultralytics import YOLO
import os
import json
import torch 

class TrafficLightDetector:
    def __init__(self, output_dir="output", model_name="yolov8n.pt"):
        """Initialize the traffic light object detector with YOLOv8 and Webots camera."""
        # Initialize Webots robot and camera
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.timestep)
        self.image_size = (self.camera.getWidth(), self.camera.getHeight())  # (512, 512)

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

    def run(self):
        """Main loop to capture images, detect objects, and display results."""
        while self.robot.step(self.timestep) != -1:
            # Capture image from Webots camera
            img = self.camera.getImage()
            img_data = np.frombuffer(img, np.uint8).reshape((self.camera.getHeight(), self.camera.getWidth(), 4))
            rgb_image = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR)

            # Process image with YOLOv8
            detected_objects = self.process_image(rgb_image)

            # Draw bounding boxes and labels on the image
            annotated_image = self.draw_detections(rgb_image, detected_objects)

            # Display the annotated image
            cv2.imshow("Object Detection", annotated_image)
            cv2.waitKey(1)  # Update the display

            # Save detected objects to log
            if detected_objects:
                self.objects_log.extend(detected_objects)
                self.save_log()

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
        timestamp = self.robot.getTime()

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
                        bbox[2] > self.image_size[1] or bbox[3] > self.image_size[0] or
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
    detector = TrafficLightDetector()
    try:
        detector.run()
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()