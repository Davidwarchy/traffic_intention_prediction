"""drive_robot controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot

if __name__ == "__main__":
    # create the Robot instance.
    robot = Robot()
    
    # get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())
    speed_max = 6.28
    
    motor_l = robot.getDevice("motor_1")
    motor_r = robot.getDevice("motor_2")
    camera_0 = robot.getDevice("camera_0")
    camera_1 = robot.getDevice("camera_1")
    
    motor_l.setPosition(float('inf'))
    motor_l.setVelocity(0.0) 
    motor_r.setPosition(float('inf'))
    motor_r.setVelocity(0.0) 
    
    camera_0.enable(timestep)
    camera_1.enable(timestep)
    
        # Main loop:
    # - perform simulation steps until Webots is stopping the controller
    while robot.step(timestep) != -1:
        speed_l = 0.5 * speed_max
        speed_r = 0.5 * speed_max
        
        motor_l.setVelocity(-speed_l)
        motor_r.setVelocity(-speed_r)
    
    # Enter here exit cleanup code.
    