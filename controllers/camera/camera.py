# camera.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from controller import Robot
import time
import os

def save_images_every_n_seconds(n, num_images, folder_name="images"):
    # Create a folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Initialize the Robot instance
    robot = Robot()

    # Get the time step of the current world
    timestep = int(robot.getBasicTimeStep())

    # Initialize devices
    camera = robot.getDevice("camera")

    # Enable the camera
    camera.enable(timestep)

    # Initialize counters and timers
    image_count = 0
    start_time = time.time()

    while robot.step(timestep) != -1 and image_count < num_images:
        current_time = time.time()

        if current_time - start_time >= n:
            # Capture and save the image
            img = camera.getImage()
            img_data = np.frombuffer(img, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
            img_bgr = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR)  # Convert from BGRA to BGR
            image_filename = os.path.join(folder_name, f"image_{image_count+1}.png")
            cv2.imwrite(image_filename, img_bgr)

            # Reset the timer and increment the image counter
            start_time = current_time
            image_count += 1


print("Starting image capture...")
save_images_every_n_seconds(1, 10)
print("Image capture complete.")