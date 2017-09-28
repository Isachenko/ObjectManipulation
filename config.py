#if you need change some configs do it in local_config.py

import os

VREP_EXE_PATH = '/Users/Isaac/V-REP_PRO_EDU_V3_4_0_Mac/vrep.app/Contents/MacOS/wrong'
MAIN_SCRIPT_PATH = os.path.abspath(".")
VREP_SCENE_PATH = MAIN_SCRIPT_PATH + '/scenes/uarmGripper.ttt'
VREP_API_PATH = MAIN_SCRIPT_PATH + '/vrep_api'

load_model = False
max_episode_length = 500
gamma = .99  # discount rate for advantage estimation and reward discounting
model_path = './model'
frames_path = './frames'
num_workers = 1

from local_config import *
print(VREP_EXE_PATH)