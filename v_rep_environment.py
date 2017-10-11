import sys
import time
import shlex
import subprocess

import numpy as np

from config import *

sys.path.append(VREP_API_PATH)
import vrep

FIRST_VREP_PORT = 19997 #port for first V_REP connection

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
        self.episode_length = MAX_EPISODE_LENGTH
        self.current_step = 0
        self.port = port

        headless = ""
        if VREP_HEADLESS:
            headless = " -h"
        xvbf = ""
        if PEREGRINE:
            xvbf = 'xvfb-run -d --server-num=1 -s "-screen 0 640x480x24" '
        bash_command = xvbf + VREP_EXE_PATH + headless + ' -gREMOTEAPISERVERSERVICE_' + str(port) + '_FALSE_TRUE ' + VREP_SCENE_PATH
        args = shlex.split(bash_command)
        print(bash_command)
        self.vrep_process = subprocess.Popen(args)
        print("sleep")
        time.sleep(8)
        print("woke up")

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
        err, self.uarm_side_camera_handle = vrep.simxGetObjectHandle(self.connection_id, 'side_camera',
                                                                vrep.simx_opmode_blocking)
        #print(self.uarm_camera_handle)

        err, self.target_object_handle = vrep.simxGetObjectHandle(self.connection_id, 'Target', vrep.simx_opmode_blocking)
        err, self.position_object_handle = vrep.simxGetObjectHandle(self.connection_id, 'Position',
                                                                  vrep.simx_opmode_blocking)
        #print(self.target_object_handle)
        print("Got handles")

    def reset(self):
        return_code = vrep.simxStopSimulation(self.connection_id, vrep.simx_opmode_blocking)
        self.current_step = 0
        time.sleep(0.1)
        #print("Environment has been reset")

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
        vrep.simxGetVisionSensorImage(self.connection_id, self.uarm_camera_handle, 1, vrep.simx_opmode_streaming)
        vrep.simxGetVisionSensorImage(self.connection_id, self.uarm_side_camera_handle, 1, vrep.simx_opmode_streaming)
        vrep.simxGetObjectVelocity(self.connection_id, self.target_object_handle, vrep.simx_opmode_streaming)
        vrep.simxGetObjectPosition(self.connection_id, self.target_object_handle, self.position_object_handle, vrep.simx_opmode_streaming)


        #print("Env has been started")
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

        err, res, image = vrep.simxGetVisionSensorImage(self.connection_id, self.uarm_camera_handle, 1, vrep.simx_opmode_buffer)
        err, res, side_image = vrep.simxGetVisionSensorImage(self.connection_id, self.uarm_side_camera_handle, 1, vrep.simx_opmode_buffer)

        #print("just get image")
        #print(res, len(image), image)
        if image == []:
            image = np.zeros([84*84], dtype=np.uint8)
        else:
            image = np.array(image, dtype=np.uint8)

        if side_image == []:
            side_image = np.zeros([84*84], dtype=np.uint8)
        else:
            side_image = np.array(side_image, dtype=np.uint8)

        #print("image: ")
        #print(image)


        state = VrepState()
        state.joints = joints_positions
        state.image = image
        state.side_image = side_image

        #print("image after: ", image)

        return state

    def get_reward(self):
        err, linear_v, ang_v = vrep.simxGetObjectVelocity(self.connection_id, self.target_object_handle, vrep.simx_opmode_buffer)
        reward = round(np.linalg.norm(np.array(linear_v)), 2)
        #print(reward)
        return reward

    def get_reward_distance(self):
        return_code, position = vrep.simxGetObjectPosition(self.connection_id, self.target_object_handle, self.position_object_handle,
                                                           vrep.simx_opmode_buffer)
        reward = 0
        for i in position:
            reward += i * i
        #reward = 1/(reward*1000)

        #print(reward)

        return reward

    def get_reward_1(self):
        err, linear_v, ang_v = vrep.simxGetObjectVelocity(self.connection_id, self.target_object_handle, vrep.simx_opmode_buffer)
        reward = round(np.linalg.norm(np.array(linear_v)), 2) - 0.00005
        #print(reward)
        return reward

    def get_reward_for_left(self):
        return_code, current_pos = vrep.simxGetJointPosition(self.connection_id, self.uarm_motor1_handle, vrep.simx_opmode_buffer)
        reward = current_pos
        #print(reward)
        return reward

    def make_action_continuous(self,value):
        if self.current_step < self.episode_length:
            value = (value - 0.5)*0.2
            #print("a:", value)
            actions = [self.rotate_clockwise_continuous, self.rotate_front_continuous, self.rotate_up_continuous]
            for i, action in enumerate(actions):
                action(value[i])

            vrep.simxSynchronousTrigger(self.connection_id)
            self.current_step += 1

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

    def rotate_clockwise_continuous(self,value):
        # print("clockwise")
        return_code, current_pos = vrep.simxGetJointPosition(self.connection_id, self.uarm_motor1_handle,
                                                             vrep.simx_opmode_buffer)
        # print(current_pos)
        new_pos = current_pos + value
        #print("RCW_C", new_pos)
        return_code = vrep.simxSetJointTargetPosition(self.connection_id, self.uarm_motor1_handle, new_pos,
                                                      vrep.simx_opmode_oneshot)

    def rotate_clockwise(self):
        # print("clockwise")
        return_code, current_pos = vrep.simxGetJointPosition(self.connection_id, self.uarm_motor1_handle, vrep.simx_opmode_buffer)
        # print(current_pos)
        new_pos = current_pos - self.eps
        #print("RW", new_pos)
        return_code = vrep.simxSetJointTargetPosition(self.connection_id, self.uarm_motor1_handle, new_pos, vrep.simx_opmode_oneshot)


    def rotate_counter_clockwise(self):
        # print("counter_clockwise")
        return_code, current_pos = vrep.simxGetJointPosition(self.connection_id, self.uarm_motor1_handle, vrep.simx_opmode_buffer)
        # print(current_pos)
        new_pos = current_pos + self.eps
        #print("RCW", new_pos)
        return_code = vrep.simxSetJointTargetPosition(self.connection_id, self.uarm_motor1_handle, new_pos, vrep.simx_opmode_oneshot)

    def rotate_front_continuous(self,value):
        # print("front")
        return_code, current_pos = vrep.simxGetJointPosition(self.connection_id, self.uarm_motor2_handle, vrep.simx_opmode_buffer)
        # print(current_pos)
        new_pos = current_pos + value
        #if (new_pos < 0.2):
        #    new_pos = 0.2

        #return_code, down_pos = vrep.simxGetJointPosition(self.connection_id, self.uarm_motor3_handle, vrep.simx_opmode_buffer)

        #if (current_pos < 0.9) and (down_pos > 2.4):
        #    new_pos = 0.9
        retur_ncode = vrep.simxSetJointTargetPosition(self.connection_id, self.uarm_motor2_handle, new_pos, vrep.simx_opmode_oneshot)

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
        #print("RF", new_pos)
        return_code = vrep.simxSetJointTargetPosition(self.connection_id, self.uarm_motor2_handle, new_pos, vrep.simx_opmode_oneshot)

    def rotate_back(self):
        # print("back")
        return_code, current_pos = vrep.simxGetJointPosition(self.connection_id, self.uarm_motor2_handle, vrep.simx_opmode_buffer)
        # print(current_pos)
        new_pos = current_pos + self.eps
        if (new_pos > 1.9):
            new_pos = 1.9
        #print("RB", new_pos)
        return_code = vrep.simxSetJointTargetPosition(self.connection_id, self.uarm_motor2_handle, new_pos, vrep.simx_opmode_oneshot)

    def rotate_up_continuous(self, value):
        # print("up")
        return_code, current_pos = vrep.simxGetJointPosition(self.connection_id, self.uarm_motor3_handle, vrep.simx_opmode_buffer)
        # print(current_pos)
        new_pos = current_pos + value
        #if (new_pos < 0.13):
        #    new_pos = 0.13
        retur_ncode = vrep.simxSetJointTargetPosition(self.connection_id, self.uarm_motor3_handle, new_pos, vrep.simx_opmode_oneshot)

    def rotate_up(self):
        # print("up")
        return_code, current_pos = vrep.simxGetJointPosition(self.connection_id, self.uarm_motor3_handle, vrep.simx_opmode_buffer)
        # print(current_pos)
        new_pos = current_pos - self.eps
        if (new_pos < 0.13):
            new_pos = 0.13
        #print("RU", new_pos)
        return_code = vrep.simxSetJointTargetPosition(self.connection_id, self.uarm_motor3_handle, new_pos, vrep.simx_opmode_oneshot)

    def rotate_down(self):
        # print("down")
        return_code, current_pos = vrep.simxGetJointPosition(self.connection_id, self.uarm_motor3_handle, vrep.simx_opmode_buffer)
        return_code, front_pos = vrep.simxGetJointPosition(self.connection_id, self.uarm_motor2_handle, vrep.simx_opmode_buffer)
        # print(current_pos)
        new_pos = current_pos + self.eps
        if (front_pos < 0.9) and (current_pos > 2.4):
            new_pos = 2.4
        #print("RD", new_pos)
        return_code = vrep.simxSetJointTargetPosition(self.connection_id, self.uarm_motor3_handle, new_pos, vrep.simx_opmode_oneshot)

    def __del__(self):
        self.disconnect_from_vrep()
        self.vrep_process.terminate()


class VrepState():
    def __init__(self):
        self.joints = [0,0,0,0,0,0]
        self.image = []

