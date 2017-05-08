import vrep
import sys
import time
import math
import random
import tensorflow as tf
print(tf.__version__)

eps = 0.1

vrep.simxFinish(-1) # just in case, close all opened connections

contiousID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5) # Connect to V-REP
if contiousID!=-1:
    print ('Connected to continuous remote API server service')
else:
    print('Connection not succesfull')
    sys.exit('could not connect')



def open_gripper():
    return_code = vrep.simxSetJointTargetVelocity(clientID, uarmGripper_motor_handle1, 0.02, vrep.simx_opmode_streaming)

def close_gripper():
    return_code = vrep.simxSetJointTargetVelocity(clientID, uarmGripper_motor_handle1, -0.02, vrep.simx_opmode_streaming)


def rotate_clockwise():
    print("clockwise")
    return_code, current_pos = vrep.simxGetJointPosition(clientID, uarm_motor1_handle, vrep.simx_opmode_buffer)
    print(current_pos)
    new_pos = current_pos - eps
    retur_ncode = vrep.simxSetJointTargetPosition(clientID, uarm_motor1_handle, new_pos, vrep.simx_opmode_oneshot)

def rotate_counter_clockwise():
    print("counter_clockwise")
    return_code, current_pos = vrep.simxGetJointPosition(clientID, uarm_motor1_handle, vrep.simx_opmode_buffer)
    print(current_pos)
    new_pos = current_pos + eps
    retur_ncode = vrep.simxSetJointTargetPosition(clientID, uarm_motor1_handle, new_pos, vrep.simx_opmode_oneshot)

def rotate_front():
    print("front")
    return_code, current_pos = vrep.simxGetJointPosition(clientID, uarm_motor2_handle, vrep.simx_opmode_buffer)
    print(current_pos)
    new_pos = current_pos - eps
    if (new_pos < 0.2):
        new_pos = 0.2

    return_code, down_pos = vrep.simxGetJointPosition(clientID, uarm_motor3_handle, vrep.simx_opmode_buffer)

    if (current_pos < 0.9) and (down_pos > 2.4):
        new_pos = 0.9
    retur_ncode = vrep.simxSetJointTargetPosition(clientID, uarm_motor2_handle, new_pos, vrep.simx_opmode_oneshot)

def rotate_back():
    print("back")
    return_code, current_pos = vrep.simxGetJointPosition(clientID, uarm_motor2_handle, vrep.simx_opmode_buffer)
    print(current_pos)
    new_pos = current_pos + eps
    if (new_pos > 1.9):
        new_pos = 1.9
    retur_ncode = vrep.simxSetJointTargetPosition(clientID, uarm_motor2_handle, new_pos, vrep.simx_opmode_oneshot)

def rotate_up():
    print("up")
    return_code, current_pos = vrep.simxGetJointPosition(clientID, uarm_motor3_handle, vrep.simx_opmode_buffer)
    print(current_pos)
    new_pos = current_pos - eps
    if (new_pos < 0.13):
        new_pos = 0.13
    retur_ncode = vrep.simxSetJointTargetPosition(clientID, uarm_motor3_handle, new_pos, vrep.simx_opmode_oneshot)

def rotate_down():
    print("down")
    return_code, current_pos = vrep.simxGetJointPosition(clientID, uarm_motor3_handle, vrep.simx_opmode_buffer)
    return_code, front_pos = vrep.simxGetJointPosition(clientID, uarm_motor2_handle, vrep.simx_opmode_buffer)
    print(current_pos)
    new_pos = current_pos + eps
    if (front_pos < 0.9) and (current_pos > 2.4):
        new_pos = 2.4
    retur_ncode = vrep.simxSetJointTargetPosition(clientID, uarm_motor3_handle, new_pos, vrep.simx_opmode_oneshot)

def get_sphere_position():
    returnCode, position = vrep.simxGetObjectPosition(clientID, sphere_handle, -1, vrep.simx_opmode_streaming)
    return position

actions = [rotate_clockwise, rotate_counter_clockwise, rotate_front, rotate_back, rotate_up, rotate_down, open_gripper, close_gripper]
#actions = [rotate_clockwise, rotate_back, rotate_up]
#actions = [rotate_counter_clockwise, rotate_front, rotate_down]
#actions = [rotate_down]



epochs_number = 10
max_t = 100
for epoch in range(epochs_number):
    return_code = vrep.simxStartSimulation(contiousID, vrep.simx_opmode_oneshot)
    time.sleep(2)
    clientID=vrep.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to V-REP

    if clientID!=-1:
        print ('Connected to remote API server')
    else:
        print('Connection not succesfull')
        return_code = vrep.simxStopSimulation(contiousID, vrep.simx_opmode_oneshot)
        vrep.simxFinish(contiousID)
        print('Simulation finished')
        sys.exit('could not connect')

    print("before getting hanles")
    # Move one joint
    error_code, uarm_motor1_handle = vrep.simxGetObjectHandle(clientID, 'uarm_motor1', vrep.simx_opmode_blocking) # Position from 0 to 3.14
    print("after getting hanle 1")

    error_code, uarm_motor2_handle = vrep.simxGetObjectHandle(clientID, 'uarm_motor2', vrep.simx_opmode_blocking)
    print("after getting hanle 2")
    print(error_code)


    error_code, uarm_motor3_handle = vrep.simxGetObjectHandle(clientID, 'uarm_motor3', vrep.simx_opmode_blocking)
    print("after getting hanle 3")

    error_code, uarm_motor4_handle = vrep.simxGetObjectHandle(clientID, 'uarm_motor4', vrep.simx_opmode_blocking)
    print("after getting hanle 4")


    error_code, uarmGripper_motor_handle1 = vrep.simxGetObjectHandle(clientID, 'uarmGripper_motor1Method2', vrep.simx_opmode_blocking)
    print("after getting hanle 5")

    error_code, uarmGripper_motor_handle2 = vrep.simxGetObjectHandle(clientID, 'uarmGripper_motor2Method2', vrep.simx_opmode_blocking)
    #error_code, sphere_handle = vrep.simxGetObjectHandle(clientID, 'Sphere', vrep.simx_opmode_blocking)
    print("after getting hanles")

    # Get joint position first time call
    return_code, position = vrep.simxGetJointPosition(clientID, uarm_motor1_handle, vrep.simx_opmode_streaming)
    return_code, position = vrep.simxGetJointPosition(clientID, uarm_motor2_handle, vrep.simx_opmode_streaming)
    return_code, position = vrep.simxGetJointPosition(clientID, uarm_motor3_handle, vrep.simx_opmode_streaming)
    return_code, position = vrep.simxGetJointPosition(clientID, uarm_motor4_handle, vrep.simx_opmode_streaming)
    return_code, position = vrep.simxGetJointPosition(clientID, uarmGripper_motor_handle1, vrep.simx_opmode_streaming)
    return_code, position = vrep.simxGetJointPosition(clientID, uarmGripper_motor_handle2, vrep.simx_opmode_streaming)
    #returnCode, position = vrep.simxGetObjectPosition(clientID, sphere_handle, -1, vrep.simx_opmode_streaming)
    print("after getting positions")


    for t in range(0, max_t):
        time.sleep(0.05)
        action = random.choice(actions)
        action()

    vrep.simxFinish(clientID)
    return_code = vrep.simxStopSimulation(contiousID, vrep.simx_opmode_oneshot)
    print("simulation finished")
    time.sleep(4)


vrep.simxFinish(contiousID)
