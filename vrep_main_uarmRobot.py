import vrep
import sys

vrep.simxFinish(-1) # just in case, close all opened connections
clientID=vrep.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to V-REP

if clientID!=-1:
    print ('Connected to remote API server')
else:
    print('Connection not succesfull')
    sys.exit('could not connect')

# Move one joint
errorCode, joint1_handle = vrep.simxGetObjectHandle(clientID,'uarm_motor1', vrep.simx_opmode_blocking)
returncode = vrep.simxSetJointTargetPosition(clientID, joint1_handle, 10, vrep.simx_opmode_oneshot)