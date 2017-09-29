#if you need change some configs do it in local_config.py

import os
MAIN_SCRIPT_PATH = os.path.abspath(".")

#vrep related params
VREP_EXE_PATH = '/Users/Isaac/V-REP_PRO_EDU_V3_4_0_Mac/vrep.app/Contents/MacOS/wrong'
VREP_SCENE_PATH = MAIN_SCRIPT_PATH + '/scenes/uarmGripper.ttt'
VREP_API_PATH = MAIN_SCRIPT_PATH + '/vrep_api'
VREP_HEADLESS = True

load_model = False
MAX_EPISODE_LENGTH = 200
gamma = .99  # discount rate for advantage estimation and reward discounting
num_workers = 1

#statistics save params
model_path = './model'
frames_path = './frames'
IMAGE_SAVE_TIME_STEP = 10
STATISTICS_SAVE_TIME_STEP = 10
MODEL_SAVE_TIME_STEP = 100

from local_config import *
print(VREP_EXE_PATH)