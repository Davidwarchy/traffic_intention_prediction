from vehicle import Driver
from controller import Keyboard

def main():
    # Initialize driver
    driver = Driver()
    
    # Set constant speed to move straight
    driver.setCruisingSpeed(50.0)  # 50 km/h
    driver.setSteeringAngle(0.0)   # Straight direction
    
    # Initialize keyboard
    keyboard = Keyboard()
    keyboard.enable(50)
    
    print("Vehicle moving straight forward at 50 km/h")
    print("Press [DOWN] to stop")
    
    while driver.step() != -1:
        # Check for stop command
        if keyboard.getKey() == Keyboard.DOWN:
            driver.setCruisingSpeed(0.0)
            print("Vehicle stopped")
            break

if __name__ == "__main__":
    main()