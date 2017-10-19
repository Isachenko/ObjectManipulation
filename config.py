#if you need change some configs do it in local_config.py

import os
import datetime
import main
import sys
MAIN_SCRIPT_PATH = os.path.abspath(".")

scenes = {  0 : "/scenes/uarm_gripper.ttt",
            1 : "/scenes/uarm_gripper_distance.ttt",
            2 : "/scenes/uarm_gripper.ttt",
            3 : "/scenes/uarm_gripper_big.ttt"
    }
#vrep related params
VREP_EXE_PATH = '/Users/Isaac/V-REP_PRO_EDU_V3_4_0_Mac/vrep.app/Contents/MacOS/wrong'
if len(sys.argv) > 1:
    s = int(sys.argv[2])
    SCENE_PATH = scenes[s]

else:
    SCENE_PATH = '/scenes/uarm_gripper.ttt'
VREP_SCENE_PATH = MAIN_SCRIPT_PATH + SCENE_PATH
REWARD_FUNCTION = ""
VREP_API_PATH = MAIN_SCRIPT_PATH + '/vrep_api'
VREP_HEADLESS = True

load_model = False
MAX_EPISODE_LENGTH = 200
MAX_NUMBER_OF_EPISODES = 1000
gamma = .99  # discount rate for advantage estimation and reward discounting
num_workers = 1



#statistics save params
now = datetime.datetime.now()
results_path = "./archive/results" + now.strftime("_%d-%m-%Y_%H-%M-%S")
model_path = results_path + '/model'
frames_path = results_path + '/frames'
statistics_path = results_path + '/train_'
archive_path = './archive'
IMAGE_SAVE_TIME_STEP = 10
STATISTICS_SAVE_TIME_STEP = 10
MODEL_SAVE_TIME_STEP = 100

PEREGRINE = False

from local_config import *
print(VREP_EXE_PATH)