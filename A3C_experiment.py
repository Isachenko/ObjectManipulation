import multiprocessing
import threading
from time import sleep

import tensorflow as tf

from ac_network import AC_Network
from v_rep_environment import *
from worker import Worker

s_size = 7056  # Observations are greyscale frames of 84 * 84 * 1
a_size = 6  # clockwise/counterclockwise, up/down, back/forth


tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)

# Create a directory to save episode playback gifs to
if not os.path.exists(frames_path):
    os.makedirs(frames_path)

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
    master_network = AC_Network(s_size, a_size, 'global', None)  # Generate global network
    if num_workers == -1:
        num_workers = multiprocessing.cpu_count()  # Set workers ot number of available CPU threads
    print("Number of workers: ", num_workers)
    workers = []
    # Create worker classes
    for i in range(num_workers):
        port = FIRST_VREP_PORT + i
        env = VRepEnvironment(port)
        workers.append(Worker(env, i, s_size, a_size, trainer, model_path, global_episodes))
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model:
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
