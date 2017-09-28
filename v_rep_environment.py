import time
import numpy as np
from config import *

import sys
sys.path.append(vrep_api_path)
print(sys.path)

import vrep


class VRepEnvironment():

    def __init__(self, port):
        self.connection_id = -1
        self.uarm_motor1_handle = -1
        self.uarm_motor2_handle = -1
        self.uarm_motor3_handle = -1
        self.uarm_motor4_handle = -1
        self.uarm_gripper_motor_handle1 = -1
        self.uarm_gripper_motor_handle2 = -1
        self.uarm_camera_handle = -1
        self.eps = 0.1
        self.episode_length = 500
        self.current_step = 0
        self.port = port
        self.connect_to_vrep()
        #self.load_scene()


    def connect_to_vrep(self):

        self.connection_id = vrep.simxStart('127.0.0.1', self.port, True, True, 5000, 5)


        if self.connection_id != -1:
            print('Connected to continuous remote API server service')
        else:
            print('Connection not successful')
            raise RuntimeError('Could not connect to V-REP')

        vrep.simxSynchronous(self.connection_id, True)
        self.get_handles()

    def load_scene(self):
        vrep.simxLoadScene(self.connection_id, 'uarmGripper.ttt', 1, vrep.simx_opmode_blocking)
        print("Scene loaded")


    def disconnect_from_vrep(self):
        vrep.simxFinish(self.connection_id)

    def get_handles(self):
        print("Getting handles")
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
        self.current_step = 0
        time.sleep(0.1)
        print("Environment has been reset")

    def new_episode(self):
        self.reset()
        self.start()

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

        joints_positions.append(
            vrep.simxGetJointPosition(self.connection_id, self.uarm_motor1_handle, vrep.simx_opmode_buffer)) # counter/clockwise
        joints_positions.append(
            vrep.simxGetJointPosition(self.connection_id, self.uarm_motor2_handle, vrep.simx_opmode_buffer))
        joints_positions.append(
            vrep.simxGetJointPosition(self.connection_id, self.uarm_motor3_handle, vrep.simx_opmode_buffer))
        joints_positions.append(
            vrep.simxGetJointPosition(self.connection_id, self.uarm_motor4_handle, vrep.simx_opmode_buffer))
        joints_positions.append(
            vrep.simxGetJointPosition(self.connection_id, self.uarm_gripper_motor_handle1, vrep.simx_opmode_buffer))
        joints_positions.append(
            vrep.simxGetJointPosition(self.connection_id, self.uarm_gripper_motor_handle2, vrep.simx_opmode_buffer))

        err, res, image = vrep.simxGetVisionSensorImage(self.connection_id, self.uarm_camera_handle, 256, vrep.simx_opmode_buffer)
        if image == []:
            image = np.zeros([84*84])
        else:
            #print("image before: ", image)
            image = np.asarray(image)
            #print("image after: ", image)

        #print("image: ")

        state = VrepState()
        state.joints = joints_positions
        state.image = image

        #print("image after: ", image)

        return state

    def get_reward(self):
        err, linear_v, ang_v = vrep.simxGetObjectVelocity(self.connection_id, self.sphere_handle, vrep.simx_opmode_buffer)
        reward = round(np.linalg.norm(np.array(linear_v)), 2)
        print(reward)
        return reward

    def make_action(self, action):
        if self.current_step < self.episode_length:
            act_num = action.index(True)
            actions = [self.rotate_clockwise, self.rotate_counter_clockwise, self.rotate_front, self.rotate_back,
                       self.rotate_up, self.rotate_down]

            #err = vrep.simxSetJointTargetVelocity(self.connection_id, self.uarm_gripper_motor_handle1, 0.02, vrep.simx_opmode_streaming)
            actions[act_num]()
            vrep.simxSynchronousTrigger(self.connection_id)
            self.current_step += 1

    def is_episode_finished(self):
        return (self.current_step == self.episode_length)

    def rotate_clockwise(self):
        # print("clockwise")
        return_code, current_pos = vrep.simxGetJointPosition(self.connection_id, self.uarm_motor1_handle, vrep.simx_opmode_buffer)
        # print(current_pos)
        new_pos = current_pos - self.eps
        retur_ncode = vrep.simxSetJointTargetPosition(self.connection_id, self.uarm_motor1_handle, new_pos, vrep.simx_opmode_oneshot)

    def rotate_counter_clockwise(self):
        # print("counter_clockwise")
        return_code, current_pos = vrep.simxGetJointPosition(self.connection_id, self.uarm_motor1_handle, vrep.simx_opmode_buffer)
        # print(current_pos)
        new_pos = current_pos + self.eps
        retur_ncode = vrep.simxSetJointTargetPosition(self.connection_id, self.uarm_motor1_handle, new_pos, vrep.simx_opmode_oneshot)

    def rotate_front(self):
        # print("front")
        return_code, current_pos = vrep.simxGetJointPosition(self.connection_id, self.uarm_motor2_handle, vrep.simx_opmode_buffer)
        # print(current_pos)
        new_pos = current_pos - self.eps
        if (new_pos < 0.2):
            new_pos = 0.2

        return_code, down_pos = vrep.simxGetJointPosition(self.connection_id, self.uarm_motor3_handle, vrep.simx_opmode_buffer)

        if (current_pos < 0.9) and (down_pos > 2.4):
            new_pos = 0.9
        retur_ncode = vrep.simxSetJointTargetPosition(self.connection_id, self.uarm_motor2_handle, new_pos, vrep.simx_opmode_oneshot)

    def rotate_back(self):
        # print("back")
        return_code, current_pos = vrep.simxGetJointPosition(self.connection_id, self.uarm_motor2_handle, vrep.simx_opmode_buffer)
        # print(current_pos)
        new_pos = current_pos + self.eps
        if (new_pos > 1.9):
            new_pos = 1.9
        retur_ncode = vrep.simxSetJointTargetPosition(self.connection_id, self.uarm_motor2_handle, new_pos, vrep.simx_opmode_oneshot)

    def rotate_up(self):
        # print("up")
        return_code, current_pos = vrep.simxGetJointPosition(self.connection_id, self.uarm_motor3_handle, vrep.simx_opmode_buffer)
        # print(current_pos)
        new_pos = current_pos - self.eps
        if (new_pos < 0.13):
            new_pos = 0.13
        retur_ncode = vrep.simxSetJointTargetPosition(self.connection_id, self.uarm_motor3_handle, new_pos, vrep.simx_opmode_oneshot)

    def rotate_down(self):
        # print("down")
        return_code, current_pos = vrep.simxGetJointPosition(self.connection_id, self.uarm_motor3_handle, vrep.simx_opmode_buffer)
        return_code, front_pos = vrep.simxGetJointPosition(self.connection_id, self.uarm_motor2_handle, vrep.simx_opmode_buffer)
        # print(current_pos)
        new_pos = current_pos + self.eps
        if (front_pos < 0.9) and (current_pos > 2.4):
            new_pos = 2.4
        retur_ncode = vrep.simxSetJointTargetPosition(self.connection_id, self.uarm_motor3_handle, new_pos, vrep.simx_opmode_oneshot)

    def __del__(self):
        self.disconnect_from_vrep()


class VrepState():
    def __init__(self):
        self.joints = [0,0,0,0,0,0]
        self.image = []

