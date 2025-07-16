import cv2
import numpy as np
from ultralytics import YOLO
import os
import json
import torch
from scipy.spatial import distance
from collections import OrderedDict

class TrafficLightTracker:
    def __init__(self, video_path, output_dir="output", model_name="yolov8n.pt", detection_interval=0.03, max_distance=100, iou_threshold=0.3):
        """Initialize the traffic light object tracker with YOLOv8 and video input."""
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")
        self.image_size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                          int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.timestep = 1.0 / self.fps if self.fps > 0 else 1.0  # Frame interval in seconds
        self.frame_delay = int(1000 / self.fps) if self.fps > 0 else 33  # Delay in ms for smooth playback

        # Device setup
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"TrafficLightTracker using device: {self.device}")

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

        # Detection and tracking setup
        self.detection_interval = detection_interval  # Time between detections
        self.last_detection_time = 0.0  # Track last detection time
        self.frame_count = 0  # Track frame number for timing
        self.tracked_objects = OrderedDict()  # Store tracked objects: {track_id: {'bbox': [x1,y1,x2,y2], 'centroid': [x,y], 'class_name': str, 'last_seen': float}}
        self.next_track_id = 0  # Incremental ID for new objects
        self.max_distance = max_distance  # Max centroid distance for tracking
        self.iou_threshold = iou_threshold  # IOU threshold for matching detections
        self.color_palette = self.generate_color_palette()  # Unique colors for each track ID

    def generate_color_palette(self):
        """Generate a list of unique colors for tracked objects."""
        colors = []
        for i in range(100):  # Generate enough colors for multiple objects
            hue = int((i * 137.5) % 360 / 2)  # Scale to 0–179 for OpenCV HSV
            hsv_color = np.array([[[hue, 255, 255]]], dtype=np.uint8)
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(int(c) for c in bgr_color))
        return colors

    def compute_iou(self, bbox1, bbox2):
        """Compute Intersection over Union (IOU) for two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        intersection = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0

    def update_tracked_objects(self, detected_objects):
        """Update tracked objects with new detections using centroid and IOU matching."""
        current_time = self.frame_count * self.timestep
        new_tracked_objects = OrderedDict()

        # Compute centroids for detected objects
        detected_centroids = []
        for obj in detected_objects:
            bbox = obj['bbox']
            centroid = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            detected_centroids.append({'centroid': centroid, 'bbox': bbox, 'class_name': obj['class_name'], 'confidence': obj['confidence']})

        # Create a list of matches: (track_id, detection_idx, score)
        matches = []
        for track_id, track_data in self.tracked_objects.items():
            if current_time - track_data['last_seen'] > 1.0:  # Remove tracks not seen for 1 second
                continue
            for idx, det in enumerate(detected_centroids):
                dist = distance.euclidean(track_data['centroid'], det['centroid'])
                iou = self.compute_iou(track_data['bbox'], det['bbox'])
                if dist < self.max_distance and iou > self.iou_threshold:
                    # Use a combined score to prioritize matches
                    score = iou / (dist + 1e-6)  # Avoid division by zero
                    matches.append((track_id, idx, score))

        # Sort matches by score in descending order
        matches.sort(key=lambda x: x[2], reverse=True)
        matched_detections = set()
        matched_tracks = set()

        # Assign matches
        for track_id, det_idx, _ in matches:
            if det_idx not in matched_detections and track_id not in matched_tracks:
                det = detected_centroids[det_idx]
                new_tracked_objects[track_id] = {
                    'bbox': det['bbox'],
                    'centroid': det['centroid'],
                    'class_name': det['class_name'],
                    'confidence': det['confidence'],
                    'last_seen': current_time
                }
                matched_detections.add(det_idx)
                matched_tracks.add(track_id)

        # Keep unmatched tracks if still recent
        for track_id, track_data in self.tracked_objects.items():
            if track_id not in matched_tracks and current_time - track_data['last_seen'] <= 1.0:
                new_tracked_objects[track_id] = track_data

        # Assign new track IDs to unmatched detections
        for idx, det in enumerate(detected_centroids):
            if idx not in matched_detections:
                new_tracked_objects[self.next_track_id] = {
                    'bbox': det['bbox'],
                    'centroid': det['centroid'],
                    'class_name': det['class_name'],
                    'confidence': det['confidence'],
                    'last_seen': current_time
                }
                self.next_track_id += 1

        self.tracked_objects = new_tracked_objects

        # Prepare log entries for current frame
        log_entries = []
        for track_id, track_data in self.tracked_objects.items():
            log_entries.append({
                'timestamp': current_time,
                'track_id': track_id,
                'class_name': track_data['class_name'],
                'confidence': track_data['confidence'],
                'bbox': track_data['bbox']
            })
        return log_entries

    def run(self):
        """Main loop to read video frames, detect and track objects, and display results."""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("End of video or failed to read frame")
                break

            # Convert frame to BGR (OpenCV reads in BGR by default)
            rgb_image = frame

            # Calculate current time based on frame count
            current_time = self.frame_count * self.timestep
            self.frame_count += 1

            # Perform detection only if the interval has passed
            detected_objects = []
            if current_time - self.last_detection_time >= self.detection_interval:
                detected_objects = self.process_image(rgb_image)
                self.last_detection_time = current_time

            # Update tracked objects
            log_entries = self.update_tracked_objects(detected_objects)

            # Draw bounding boxes and labels on the image
            annotated_image = self.draw_detections(rgb_image)

            # Display the annotated image
            cv2.imshow("Object Tracking", annotated_image)
            if cv2.waitKey(self.frame_delay) & 0xFF == ord('q'):  # Exit on 'q' key press
                break

            # Save detected objects to log if any were detected
            if log_entries:
                self.objects_log.extend(log_entries)
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
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
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
                        'timestamp': self.frame_count * self.timestep,
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': bbox
                    }
                    detected_objects.append(obj_data)
                except Exception as e:
                    print(f"Error processing box: {e}")
                    continue

        return detected_objects

    def draw_detections(self, image):
        """Draw bounding boxes and labels on the image for tracked objects."""
        annotated_image = image.copy()
        for track_id, obj in self.tracked_objects.items():
            bbox = obj['bbox']
            class_name = obj['class_name']
            confidence = obj['confidence']
            color = self.color_palette[track_id % len(self.color_palette)]
            cv2.rectangle(annotated_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            label = f"ID {track_id}: {class_name} {confidence:.2f}"
            cv2.putText(annotated_image, label, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
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
    """Initialize and run the traffic light tracker."""
    video_path = r"C:\Users\HP\Desktop\DSAIL\acvss\vehicles\controllers\camera\real_traffic.mp4"
    tracker = TrafficLightTracker(video_path=video_path, detection_interval=0.03)
    try:
        tracker.run()
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()