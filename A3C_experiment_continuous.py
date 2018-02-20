import multiprocessing
import threading
from time import sleep
import datetime


import tensorflow as tf
print(tf.__version__)

from ac_network_continuous import ACNetworkContinuous
from ac_network_continuous_gaussian import ACNetworkContinuousGaussian
from v_rep_environment import *
from worker_continuous import WorkerContinuous


import shutil

s_size = 7056  # Observations are greyscale frames of 84 * 84 * 1
a_size = 6  # clockwise/counterclockwise, up/down, back/forth


tf.reset_default_graph()

if not os.path.exists(results_path):
    os.makedirs(results_path)

if not os.path.exists(model_path):
    os.makedirs(model_path)

# Create a directory to save episode playback gifs to
if not os.path.exists(frames_path):
    os.makedirs(frames_path)

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
    master_network = ACNetworkContinuousGaussian(s_size, a_size, 'global', None)  # Generate global network
    if num_workers == -1:
        num_workers = multiprocessing.cpu_count()  # Set workers at number of available CPU threads
    print("Number of workers: ", num_workers)
    workers = []
    # Create worker classes
    for i in range(num_workers):
        port = FIRST_VREP_PORT + i
        env = VRepEnvironment(port)
        workers.append(WorkerContinuous(env, i, s_size, a_size, trainer, model_path, global_episodes))
    saver = tf.train.Saver(max_to_keep=1)

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
        worker_work = lambda: worker.work(MAX_EPISODE_LENGTH, gamma, sess, coord, saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        print("Thread started")
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)

#After every thread stops
#now = datetime.datetime.now()
#new_name = results_path + now.strftime("_%d-%m-%Y_%H-%M-%S")
#shutil.move(results_path, new_name)

if not os.path.exists(archive_path):
    os.makedirs(archive_path)

#shutil.move(new_name, archive_path)
