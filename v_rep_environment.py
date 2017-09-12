import vrep
import time
import numpy as np


class VRepEnvironment():

    def __init__(self):
        self.connection_id = -1
        self.uarm_motor1_handle = -1
        self.uarm_motor2_handle = -1
        self.uarm_motor3_handle = -1
        self.uarm_motor4_handle = -1
        self.uarm_gripper_motor_handle1 = -1
        self.uarm_gripper_motor_handle2 = -1
        self.uarm_camera_handle = -1

    def connect_to_vrep(self):
        vrep.simxFinish(-1)  # just in case, close all opened connections
        self.connection_id = vrep.simxStart('127.0.0.1',19997,True,True,5000,5)
        if self.connection_id != -1:
            print('Connected to continuous remote API server service')
        else:
            print('Connection not succesfull')
            raise RuntimeError('could not connect to V-REP')

        vrep.simxSynchronous(self.connection_id, True)
        self.get_handles()

    def disconnect_from_vrep(self):
        vrep.simxFinish(self.connection_id)

    def get_handles(self):
        print("Getting hanles")
        # Move one joint
        err, self.uarm_motor1_handle = vrep.simxGetObjectHandle(self.connection_id, 'uarm_motor1',
                                                                 vrep.simx_opmode_blocking)  # Position from 0 to 3.14

        err, self.uarm_motor2_handle = vrep.simxGetObjectHandle(self.connection_id, 'uarm_motor2',
                                                                 vrep.simx_opmode_blocking)

        err, self.uarm_motor3_handle = vrep.simxGetObjectHandle(self.connection_id, 'uarm_motor3',
                                                                 vrep.simx_opmode_blocking)

        err, self.uarm_motor4_handle = vrep.simxGetObjectHandle(self.connection_id, 'uarm_motor4',
                                                                 vrep.simx_opmode_blocking)

        err, self.uarm_gripper_motor_handle1 = vrep.simxGetObjectHandle(self.connection_id, 'uarmGripper_motor1Method2',
                                                                        vrep.simx_opmode_blocking)

        err, self.uarm_gripper_motor_handle2 = vrep.simxGetObjectHandle(self.connection_id, 'uarmGripper_motor2Method2',
                                                                        vrep.simx_opmode_blocking)

        err, self.uarm_camera_handle = vrep.simxGetObjectHandle(self.connection_id, 'Vision_sensor',
                                                                        vrep.simx_opmode_blocking)

        err, self.sphere_handle = vrep.simxGetObjectHandle(self.connection_id, 'Sphere', vrep.simx_opmode_blocking)
        print("Got handles")

    def reset(self):
        return_code = vrep.simxStopSimulation(self.connection_id, vrep.simx_opmode_blocking)
        time.sleep(0.1)
        print("Env has been reset")

    def start(self):
        return_code = vrep.simxStartSimulation(self.connection_id, vrep.simx_opmode_blocking)
        # Get joint position first time call
        vrep.simxGetJointPosition(self.connection_id, self.uarm_motor1_handle, vrep.simx_opmode_streaming)
        vrep.simxGetJointPosition(self.connection_id, self.uarm_motor2_handle, vrep.simx_opmode_streaming)
        vrep.simxGetJointPosition(self.connection_id, self.uarm_motor3_handle, vrep.simx_opmode_streaming)
        vrep.simxGetJointPosition(self.connection_id, self.uarm_motor4_handle, vrep.simx_opmode_streaming)
        vrep.simxGetJointPosition(self.connection_id, self.uarm_gripper_motor_handle1, vrep.simx_opmode_streaming)
        vrep.simxGetJointPosition(self.connection_id, self.uarm_gripper_motor_handle2, vrep.simx_opmode_streaming)
        vrep.simxGetVisionSensorImage(self.connection_id, self.uarm_camera_handle, 0, vrep.simx_opmode_streaming)
        vrep.simxGetObjectVelocity(self.connection_id, self.sphere_handle, vrep.simx_opmode_streaming)

        print("Env has been started")
        # returnCode, position = vrep.simxGetObjectPosition(self.connection_id, sphere_handle, -1, vrep.simx_opmode_streaming)

    def get_state(self):
        joints_positions = []

        joints_positions.append(vrep.simxGetJointPosition(self.connection_id, self.uarm_motor1_handle, vrep.simx_opmode_buffer)) # counter/clockwise
        joints_positions.append(vrep.simxGetJointPosition(self.connection_id, self.uarm_motor2_handle, vrep.simx_opmode_buffer))
        joints_positions.append(vrep.simxGetJointPosition(self.connection_id, self.uarm_motor3_handle, vrep.simx_opmode_buffer))
        joints_positions.append(vrep.simxGetJointPosition(self.connection_id, self.uarm_motor4_handle, vrep.simx_opmode_buffer))
        joints_positions.append(vrep.simxGetJointPosition(self.connection_id, self.uarm_gripper_motor_handle1, vrep.simx_opmode_buffer))
        joints_positions.append(vrep.simxGetJointPosition(self.connection_id, self.uarm_gripper_motor_handle2, vrep.simx_opmode_buffer))

        image = vrep.simxGetVisionSensorImage(self.connection_id, self.uarm_camera_handle, 0, vrep.simx_opmode_buffer)

        #print(image)

        state = VrepState()
        state.joints = joints_positions
        state.image = image

        return state

    def get_reward(self):
        err, linear_v, ang_v = vrep.simxGetObjectVelocity(self.connection_id, self.sphere_handle, vrep.simx_opmode_buffer)
        reward = round(np.linalg.norm(np.array(linear_v)), 2)
        print(reward)
        return reward

    def make_action(self, action):
        err = vrep.simxSetJointTargetVelocity(self.connection_id, self.uarm_gripper_motor_handle1, 0.02, vrep.simx_opmode_streaming)
        vrep.simxSynchronousTrigger(self.connection_id)


class VrepState():
    def __init__(self):
        self.joints = [0,0,0,0,0,0]
        self.image = []

