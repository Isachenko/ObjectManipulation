#This file contain local setting for running programm
#Change it once for yourself and
#do not push this file

vrep_exec_path = '../V-REP_PRO_EDU_V3_4_0_Mac/vrep.app/Contents/MacOS/vrep'
vrep_scene_path = '../../../../ObjectManipulation/uarmGripper.ttt'
load_model = False
max_episode_length = 500
gamma = .99  # discount rate for advantage estimation and reward discounting
model_path = './model'
num_workers = 1