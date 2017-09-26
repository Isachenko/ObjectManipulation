import tensorflow as tf
import os
from ac_network import AC_Network
import multiprocessing
import threading
from time import sleep
from worker import Worker
from v_rep_environment import *
import subprocess
import shlex

max_episode_length = 500
gamma = .99  # discount rate for advantage estimation and reward discounting
s_size = 7056  # Observations are greyscale frames of 84 * 84 * 1
a_size = 6  # clockwise/counterclockwise, up/down, back/forth
load_model = True
model_path = './model'
vrep_exec_path = '../V-REP_PRO_EDU_V3_4_0_Mac/vrep.app/Contents/MacOS/vrep'
vrep_scene_path = '../../../../ObjectManipulation/uarmGripper.ttt'

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)

# Create a directory to save episode playback gifs to
if not os.path.exists('./frames'):
    os.makedirs('./frames')

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
    master_network = AC_Network(s_size, a_size, 'global', None)  # Generate global network
    num_workers = 2  # multiprocessing.cpu_count()  # Set workers ot number of available CPU threads
    workers = []
    # Create worker classes
    vrep.simxFinish(-1)  # just in case, close all opened connections



    for i in range(num_workers):
        port = 19997 + i
        bash_command = vrep_exec_path + ' -h -gREMOTEAPISERVERSERVICE_' + str(port) + '_FALSE_TRUE ' + vrep_scene_path
        args = shlex.split(bash_command)
        print(bash_command)
        print(args)
        p = subprocess.Popen(args)
        print("sleep")
        sleep(5)
        print("woke up")
        env = VRepEnvironment(port) #Several enviroments??
        workers.append(Worker(env, i, s_size, a_size, trainer, model_path, global_episodes))
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length, gamma, sess, coord, saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        print("Thread started")
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)
