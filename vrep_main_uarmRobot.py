import vrep
import sys
import time

vrep.simxFinish(-1) # just in case, close all opened connections
clientID=vrep.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to V-REP

if clientID!=-1:
    print ('Connected to remote API server')
else:
    print('Connection not succesfull')
    sys.exit('could not connect')

# Move one joint
errorCode, uarm_motor1_handle = vrep.simxGetObjectHandle(clientID,'uarm_motor1', vrep.simx_opmode_blocking) # Position from 0 to 3.14
errorCode, uarm_motor2_handle = vrep.simxGetObjectHandle(clientID,'uarm_motor2', vrep.simx_opmode_blocking)
errorCode, uarm_motor3_handle = vrep.simxGetObjectHandle(clientID,'uarm_motor3', vrep.simx_opmode_blocking)
errorCode, uarm_motor4_handle = vrep.simxGetObjectHandle(clientID,'uarm_motor4', vrep.simx_opmode_blocking)
errorCode, uarmGripper_motor_handle1 = vrep.simxGetObjectHandle(clientID,'uarmGripper_motor1Method2', vrep.simx_opmode_blocking)
errorCode, uarmGripper_motor_handle2 = vrep.simxGetObjectHandle(clientID,'uarmGripper_motor2Method2', vrep.simx_opmode_blocking)



# Get joint position first time call
returnCode, position = vrep.simxGetJointPosition(clientID, uarm_motor1_handle, vrep.simx_opmode_streaming)
returnCode, position = vrep.simxGetJointPosition(clientID, uarm_motor2_handle, vrep.simx_opmode_streaming)
returnCode, position = vrep.simxGetJointPosition(clientID, uarm_motor3_handle, vrep.simx_opmode_streaming)
returnCode, position = vrep.simxGetJointPosition(clientID, uarm_motor4_handle, vrep.simx_opmode_streaming)
returnCode, position = vrep.simxGetJointPosition(clientID, uarmGripper_motor_handle1, vrep.simx_opmode_streaming)
returnCode, position = vrep.simxGetJointPosition(clientID, uarmGripper_motor_handle2, vrep.simx_opmode_streaming)


returnCode = vrep.simxSetJointTargetVelocity(clientID, uarmGripper_motor_handle1, 1, vrep.simx_opmode_oneshot)

print(returnCode)




     #     returncode = vrep.simxSetJointTargetPosition(clientID, uarm_motor1_handle, x, vrep.simx_opmode_oneshot)
#     returncode = vrep.simxSetJointTargetPosition(clientID, uarm_motor2_handle, x, vrep.simx_opmode_oneshot)
#     returncode = vrep.simxSetJointTargetPosition(clientID, uarm_motor3_handle, x, vrep.simx_opmode_oneshot)
#     returncode = vrep.simxSetJointTargetPosition(clientID, uarm_motor4_handle, x, vrep.simx_opmode_oneshot)
#     returnCode, position = vrep.simxSetJointTargetVelocity(clientID, uarmGripper_motor_handle1, x,vrep.simx_opmode_buffer)
#     returnCode, position = vrep.simxSetJointTargetVelocity(clientID, uarmGripper_motor_handle2, x,vrep.simx_opmode_buffer)

    #
#     time.sleep(1)
#
#     returnCode, position1 = vrep.simxGetJointPosition(clientID, uarm_motor1_handle, vrep.simx_opmode_buffer)
#     returnCode, position2 = vrep.simxGetJointPosition(clientID, uarm_motor2_handle, vrep.simx_opmode_buffer)
#     returnCode, position3 = vrep.simxGetJointPosition(clientID, uarm_motor3_handle, vrep.simx_opmode_buffer)
#     returnCode, position4 = vrep.simxGetJointPosition(clientID, uarm_motor4_handle, vrep.simx_opmode_buffer)
#
#     print("X Value", x, "uarm_motor1", position1, "uarm_motor2", position2, "uarm_motor2", position3, "uarm_motor4", position1)