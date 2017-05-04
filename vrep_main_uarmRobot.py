import vrep
import sys
import time
import math
import random

eps = 0.2

vrep.simxFinish(-1) # just in case, close all opened connections
clientID=vrep.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to V-REP

if clientID!=-1:
    print ('Connected to remote API server')
else:
    print('Connection not succesfull')
    sys.exit('could not connect')

# Move one joint
error_code, uarm_motor1_handle = vrep.simxGetObjectHandle(clientID, 'uarm_motor1', vrep.simx_opmode_blocking) # Position from 0 to 3.14
error_code, uarm_motor2_handle = vrep.simxGetObjectHandle(clientID, 'uarm_motor2', vrep.simx_opmode_blocking)
error_code, uarm_motor3_handle = vrep.simxGetObjectHandle(clientID, 'uarm_motor3', vrep.simx_opmode_blocking)
error_code, uarm_motor4_handle = vrep.simxGetObjectHandle(clientID, 'uarm_motor4', vrep.simx_opmode_blocking)
error_code, uarmGripper_motor_handle1 = vrep.simxGetObjectHandle(clientID, 'uarmGripper_motor1Method2', vrep.simx_opmode_blocking)
error_code, uarmGripper_motor_handle2 = vrep.simxGetObjectHandle(clientID, 'uarmGripper_motor2Method2', vrep.simx_opmode_blocking)
error_code, sphere_handle = vrep.simxGetObjectHandle(clientID, 'Sphere', vrep.simx_opmode_blocking)

# Get joint position first time call
return_code, position = vrep.simxGetJointPosition(clientID, uarm_motor1_handle, vrep.simx_opmode_streaming)
return_code, position = vrep.simxGetJointPosition(clientID, uarm_motor2_handle, vrep.simx_opmode_streaming)
return_code, position = vrep.simxGetJointPosition(clientID, uarm_motor3_handle, vrep.simx_opmode_streaming)
return_code, position = vrep.simxGetJointPosition(clientID, uarm_motor4_handle, vrep.simx_opmode_streaming)
return_code, position = vrep.simxGetJointPosition(clientID, uarmGripper_motor_handle1, vrep.simx_opmode_streaming)
return_code, position = vrep.simxGetJointPosition(clientID, uarmGripper_motor_handle2, vrep.simx_opmode_streaming)
returnCode, position = vrep.simxGetObjectPosition(clientID, sphere_handle, -1, vrep.simx_opmode_streaming)

def open_gripper():
    return_code = vrep.simxSetJointTargetVelocity(clientID, uarmGripper_motor_handle1, 0.02, vrep.simx_opmode_streaming)

def close_gripper():
    return_code = vrep.simxSetJointTargetVelocity(clientID, uarmGripper_motor_handle1, -0.02, vrep.simx_opmode_streaming)


def rotate_clockwise():
    return_code, current_pos = vrep.simxGetJointPosition(clientID, uarm_motor1_handle, vrep.simx_opmode_buffer)
    print(current_pos)
    new_pos = current_pos - eps
    retur_ncode = vrep.simxSetJointTargetPosition(clientID, uarm_motor1_handle, new_pos, vrep.simx_opmode_oneshot)

def rotate_counter_clockwise():
    return_code, current_pos = vrep.simxGetJointPosition(clientID, uarm_motor1_handle, vrep.simx_opmode_buffer)
    print(current_pos)
    new_pos = current_pos + eps
    retur_ncode = vrep.simxSetJointTargetPosition(clientID, uarm_motor1_handle, new_pos, vrep.simx_opmode_oneshot)

def rotate_front():
    return_code, current_pos = vrep.simxGetJointPosition(clientID, uarm_motor2_handle, vrep.simx_opmode_buffer)
    print(current_pos)
    new_pos = current_pos + eps
    retur_ncode = vrep.simxSetJointTargetPosition(clientID, uarm_motor2_handle, new_pos, vrep.simx_opmode_oneshot)

def rotate_back():
    return_code, current_pos = vrep.simxGetJointPosition(clientID, uarm_motor2_handle, vrep.simx_opmode_buffer)
    print(current_pos)
    new_pos = current_pos - eps
    retur_ncode = vrep.simxSetJointTargetPosition(clientID, uarm_motor2_handle, new_pos, vrep.simx_opmode_oneshot)

def rotate_up():
    return_code, current_pos = vrep.simxGetJointPosition(clientID, uarm_motor3_handle, vrep.simx_opmode_buffer)
    print(current_pos)
    new_pos = current_pos - eps
    retur_ncode = vrep.simxSetJointTargetPosition(clientID, uarm_motor3_handle, new_pos, vrep.simx_opmode_oneshot)

def rotate_down():
    return_code, current_pos = vrep.simxGetJointPosition(clientID, uarm_motor3_handle, vrep.simx_opmode_buffer)
    print(current_pos)
    new_pos = current_pos + eps
    retur_ncode = vrep.simxSetJointTargetPosition(clientID, uarm_motor3_handle, new_pos, vrep.simx_opmode_oneshot)

def get_sphere_position():
    returnCode, position = vrep.simxGetObjectPosition(clientID, sphere_handle, -1, vrep.simx_opmode_streaming)
    return position

actions = [rotate_clockwise, rotate_counter_clockwise, rotate_front, rotate_back, rotate_up, rotate_down, open_gripper, close_gripper]


n_steps = 50
for i in range(0, n_steps):
    time.sleep(0.5)
    action = random.choice(actions)
    action()
