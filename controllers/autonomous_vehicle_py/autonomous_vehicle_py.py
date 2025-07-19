from vehicle import Driver
from controller import Camera, Lidar, GPS, Keyboard
import math

# PID constants
KP = 0.25
KI = 0.006
KD = 2

# Line following variables
FILTER_SIZE = 3
UNKNOWN = 99999.99
PID_need_reset = False

# Device flags
enable_collision_avoidance = False
has_gps = False
has_camera = False

# Global variables
speed = 0.0
steering_angle = 0.0
manual_steering = 0
autodrive = True
gps_coords = [0.0, 0.0, 0.0]
gps_speed = 0.0

def print_help():
    print("Python - You can drive this car!")
    print("Select the 3D window and use cursor keys to:")
    print("[LEFT]/[RIGHT] - steer")
    print("[UP]/[DOWN] - accelerate/slow down")

def set_autodrive(driver, onoff):
    global autodrive
    if autodrive == onoff:
        return
    autodrive = onoff
    if autodrive:
        if has_camera:
            print("switching to auto-drive...")
        else:
            print("impossible to switch auto-drive on without camera...")
    else:
        print("switching to manual drive...")
        print("hit [A] to return to auto-drive.")

def set_speed(driver, kmh):
    global speed
    if kmh > 250.0:
        kmh = 250.0
    speed = kmh
    print(f"setting speed to {kmh} km/h")
    driver.setCruisingSpeed(kmh)

def set_steering_angle(driver, wheel_angle):
    global steering_angle
    if wheel_angle - steering_angle > 0.1:
        wheel_angle = steering_angle + 0.1
    if wheel_angle - steering_angle < -0.1:
        wheel_angle = steering_angle - 0.1
    steering_angle = wheel_angle
    if wheel_angle > 0.5:
        wheel_angle = 0.5
    elif wheel_angle < -0.5:
        wheel_angle = -0.5
    driver.setSteeringAngle(wheel_angle)

def change_manual_steer_angle(driver, inc):
    global manual_steering
    set_autodrive(driver, False)
    new_manual_steering = manual_steering + inc
    if -25.0 <= new_manual_steering <= 25.0:
        manual_steering = new_manual_steering
        set_steering_angle(driver, manual_steering * 0.02)
    if manual_steering == 0:
        print("going straight")
    else:
        print(f"turning {steering_angle:.2f} rad ({'left' if steering_angle < 0 else 'right'})")

def check_keyboard(driver, keyboard):
    key = keyboard.getKey()
    if key == Keyboard.UP:
        set_speed(driver, speed + 5.0)
    elif key == Keyboard.DOWN:
        set_speed(driver, speed - 5.0)
    elif key == Keyboard.RIGHT:
        change_manual_steer_angle(driver, 1)
    elif key == Keyboard.LEFT:
        change_manual_steer_angle(driver, -1)
    elif key == ord('A'):
        set_autodrive(driver, True)

def color_diff(a, b):
    diff = 0
    for i in range(3):
        d = a[i] - b[i]
        diff += abs(d)
    return diff

def process_camera_image(image, camera_width, camera_height, camera_fov):
    REF = [95, 187, 203]  # Yellow line color (BGR)
    sumx = 0
    pixel_count = 0
    for x in range(camera_width * camera_height):
        pixel = [image[4*x], image[4*x+1], image[4*x+2]]
        if color_diff(pixel, REF) < 30:
            sumx += x % camera_width
            pixel_count += 1
    if pixel_count == 0:
        return UNKNOWN
    return ((sumx / pixel_count / camera_width) - 0.5) * camera_fov

def filter_angle(new_value):
    global first_call, old_values
    if 'first_call' not in globals():
        globals()['first_call'] = True
        globals()['old_values'] = [0.0] * FILTER_SIZE
    if first_call or new_value == UNKNOWN:
        globals()['first_call'] = False
        globals()['old_values'] = [0.0] * FILTER_SIZE
    elif new_value != UNKNOWN:
        old_values[:-1] = old_values[1:]
        old_values[-1] = new_value
    if new_value == UNKNOWN:
        return UNKNOWN
    return sum(old_values) / FILTER_SIZE

def process_sick_data(sick_data, sick_width, sick_fov):
    HALF_AREA = 20
    sumx = 0
    collision_count = 0
    obstacle_dist = 0.0
    for x in range(sick_width // 2 - HALF_AREA, sick_width // 2 + HALF_AREA):
        range_val = sick_data[x]
        if range_val < 20.0:
            sumx += x
            collision_count += 1
            obstacle_dist += range_val
    if collision_count == 0:
        return UNKNOWN, 0.0
    obstacle_dist /= collision_count
    return ((sumx / collision_count / sick_width) - 0.5) * sick_fov, obstacle_dist

def compute_gps_speed(gps):
    global gps_coords, gps_speed
    gps_coords = gps.getValues()
    gps_speed = gps.getSpeed() * 3.6  # Convert m/s to km/h

def apply_PID(yellow_line_angle):
    global PID_need_reset, old_value, integral
    if 'old_value' not in globals():
        globals()['old_value'] = 0.0
        globals()['integral'] = 0.0
    if PID_need_reset:
        old_value = yellow_line_angle
        integral = 0.0
        globals()['PID_need_reset'] = False
    if (yellow_line_angle < 0) != (old_value < 0):
        integral = 0.0
    diff = yellow_line_angle - old_value
    if -30 < integral < 30:
        integral += yellow_line_angle
    globals()['old_value'] = yellow_line_angle
    return KP * yellow_line_angle + KI * integral + KD * diff

def main():
    global enable_collision_avoidance, has_gps, has_camera
    driver = Driver()
    
    # Check available devices
    for device in [driver.getDeviceByIndex(i) for i in range(driver.getNumberOfDevices())]:
        name = device.getName()
        if name == "Sick LMS 291":
            enable_collision_avoidance = True
        elif name == "gps":
            has_gps = True
        elif name == "camera":
            has_camera = True

    # Initialize devices
    if has_camera:
        camera = driver.getDevice("camera")
        camera.enable(50)
        camera_width = camera.getWidth()
        camera_height = camera.getHeight()
        camera_fov = camera.getFov()
    if enable_collision_avoidance:
        sick = driver.getDevice("Sick LMS 291")
        sick.enable(50)
        sick_width = sick.getHorizontalResolution()
        sick_fov = sick.getFov()
    if has_gps:
        gps = driver.getDevice("gps")
        gps.enable(50)
    
    # Initialize vehicle
    if has_camera:
        set_speed(driver, 50.0)
    driver.setHazardFlashers(True)
    driver.setDippedBeams(True)
    driver.setAntifogLights(True)
    driver.setWiperMode(Driver.SLOW)
    
    keyboard = Keyboard()
    keyboard.enable(50)
    print_help()
    
    i = 0
    while driver.step() != -1:
        if i % (50 // driver.getBasicTimeStep()) == 0:
            check_keyboard(driver, keyboard)
            camera_image = camera.getImage() if has_camera else None
            sick_data = sick.getRangeImage() if enable_collision_avoidance else None
            
            if autodrive and has_camera:
                yellow_line_angle = filter_angle(process_camera_image(camera_image, camera_width, camera_height, camera_fov))
                obstacle_angle, obstacle_dist = process_sick_data(sick_data, sick_width, sick_fov) if enable_collision_avoidance else (UNKNOWN, 0.0)
                
                if enable_collision_avoidance and obstacle_angle != UNKNOWN:
                    driver.setBrakeIntensity(0.0)
                    obstacle_steering = steering_angle
                    if 0.0 < obstacle_angle < 0.4:
                        obstacle_steering += (obstacle_angle - 0.25) / obstacle_dist
                    elif obstacle_angle > -0.4:
                        obstacle_steering += (obstacle_angle + 0.25) / obstacle_dist
                    steer = steering_angle
                    if yellow_line_angle != UNKNOWN:
                        line_following_steering = apply_PID(yellow_line_angle)
                        if obstacle_steering > 0 and line_following_steering > 0:
                            steer = max(obstacle_steering, line_following_steering)
                        elif obstacle_steering < 0 and line_following_steering < 0:
                            steer = min(obstacle_steering, line_following_steering)
                    else:
                        globals()['PID_need_reset'] = True
                    set_steering_angle(driver, steer)
                elif yellow_line_angle != UNKNOWN:
                    driver.setBrakeIntensity(0.0)
                    set_steering_angle(driver, apply_PID(yellow_line_angle))
                else:
                    driver.setBrakeIntensity(0.4)
                    globals()['PID_need_reset'] = True
            
            if has_gps:
                compute_gps_speed(gps)
        
        i += 1

if __name__ == "__main__":
    main()