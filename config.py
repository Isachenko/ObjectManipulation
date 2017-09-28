#This file contain local setting for running programm
#Change it once for yourself and
#do not push this file

vrep_exec_path = '/Users/Isaac/V-REP_PRO_EDU_V3_4_0_Mac/vrep.app/Contents/MacOS/vrep'
vrep_scene_path = '/Users/Isaac/ObjectManipulation/scenes/uarmGripper.ttt'
vrep_api_path = './vrep_api'
#vrep_api_path = '/Users/Isaac/V-REP_PRO_EDU_V3_4_0_Mac/programming/remoteApiBindings/python/python'
#vrep_lib_path = '/Users/Isaac/V-REP_PRO_EDU_V3_4_0_Mac/programming/remoteApiBindings/lib/lib'

load_model = False
max_episode_length = 500
gamma = .99  # discount rate for advantage estimation and reward discounting
model_path = './model'
num_workers = 1