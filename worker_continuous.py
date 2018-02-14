from ac_network_continuous import ACNetworkContinuous
from ac_network_continuous_gaussian import ACNetworkContinuousGaussian
from utils.helper import *
from utils.utils import *
from config import *
import random
import numpy as np
import scipy.misc


class WorkerContinuous():
    def __init__(self, game, name, s_size, a_size, trainer, model_path, global_episodes):
        self.a_size = a_size
        self.name = "c_worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter(statistics_path + str(self.number))

        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = ACNetworkContinuousGaussian(s_size, a_size, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)

        self.actions = self.actions = np.identity(a_size, dtype=bool).tolist()
        self.env = game
        self.exp_decay = 0



    def train(self, rollout, sess, gamma, bootstrap_value, sigma):
        rollout = np.array(rollout)
        #print("rollout:", rollout)
        observations = rollout[:, 0]
        actions = np.vstack(np.array(rollout[:, 1]))
        rewards = rollout[:, 2]
        next_observations = rollout[:, 3]
        values = rollout[:, 5]

        # print("observations:", observations)
        # print("observations:", np.vstack(observations))
        # print("actions:", actions)
        # print("rewards:", rewards)
        # print("values:", values)

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        #print("discounted_rewards: ", discounted_rewards)
        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.inputs: np.vstack(observations),
                     self.local_AC.actions: actions,
                     self.local_AC.advantages: advantages,
                     self.local_AC.state_in[0]: self.batch_rnn_state[0],
                     self.local_AC.state_in[1]: self.batch_rnn_state[1],
                     self.local_AC.policy_sigma: sigma}
        v_l, p_l, e_l, g_n, v_n, self.batch_rnn_state, _ = sess.run([self.local_AC.value_loss,
                                                                     self.local_AC.policy_loss,
                                                                     self.local_AC.mean_entropy,
                                                                     self.local_AC.grad_norms,
                                                                     self.local_AC.var_norms,
                                                                     self.local_AC.state_out,
                                                                     self.local_AC.apply_grads],
                                                                    feed_dict=feed_dict)
        #print("VL_0: ", v_l)
        #print("PL_0: ", p_l)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def work(self, max_episode_length, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        #self.episode_number = episode_count
        total_steps = 0
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default(), coord.stop_on_exception():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                episode_policy_sigma = np.ones(self.a_size)*(1/np.log(3+episode_count**2))
                #print(episode_policy_sigma)
                d = False

                self.env.new_episode()
                s = self.env.get_state().image
                s = process_frame(s)
                rnn_state = self.local_AC.state_init
                self.batch_rnn_state = rnn_state
                while self.env.is_episode_finished() == False:
                    a, v, rnn_state = sess.run(
                        [self.local_AC.A, self.local_AC.value, self.local_AC.state_out],
                        feed_dict={self.local_AC.inputs: [s],
                                   self.local_AC.state_in[0]: rnn_state[0],
                                   self.local_AC.state_in[1]: rnn_state[1],
                                   self.local_AC.policy_sigma: episode_policy_sigma})

                    #print(a)
                    #a = a[0]
                    #print(a)

                    self.env.make_action_continuous(a)

                    if len(sys.argv) > 1:
                        r = self.env.get_reward_command_line()
                    else:
                        r = self.env.get_reward()

                    d = self.env.is_episode_finished()
                    if d == False:
                        s1 = self.env.get_state().image
                        if self.number == 0:
                            side_s = self.env.get_state().side_image
                            frame = process_gif(s1, side_s)
                            episode_frames.append(frame)


                        s1 = process_frame(s1)
                    else:
                        s1 = s

                    #print("s", s)
                    episode_buffer.append([s, a, r, s1, d, v[0, 0]])
                    episode_values.append(v[0, 0])

                    episode_reward += r
                    s = s1
                    total_steps += 1
                    episode_step_count += 1

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer) == 30 and d != True and episode_step_count != max_episode_length - 1:
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        #print("time to train")
                        #print(episode_buffer.shape)
                        #print(episode_buffer[0])
                        v1 = sess.run(self.local_AC.value,
                                      feed_dict={self.local_AC.inputs: [s],
                                                 self.local_AC.state_in[0]: rnn_state[0],
                                                 self.local_AC.state_in[1]: rnn_state[1]})[0, 0]
                        #print("before train")
                        v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, v1, episode_policy_sigma)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if d == True:
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the episode buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, 0.0, episode_policy_sigma)

                # Periodically save gifs of episodes, model parameters, and summary statistics.
                #print("Episode count:",episode_count)
                if episode_count % STATISTICS_SAVE_TIME_STEP == 0 and episode_count != 0:
                    if self.number == 0 and episode_count % IMAGE_SAVE_TIME_STEP == 0:
                        time_per_step = 0.05
                        images = np.array(episode_frames)
                        make_gif(images, frames_path + '/image' + str(episode_count) + '.gif',
                                 duration=len(images) * time_per_step, true_image=True, salience=False)
                        #scipy.misc.toimage(images[10], cmin=0.0, cmax=...).save(frames_path + '/image' + str(episode_count) + '.jpg')
                        #scipy.misc.imsave(frames_path + '/image' + str(episode_count) + '.jpg', images[10])

                    if episode_count % MODEL_SAVE_TIME_STEP == 0 and self.number == 0:
                        saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                        print("Saved Model")


                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Params/Sigma', simple_value=float(episode_policy_sigma[0]))
                    summary.value.add(tag='Params/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()

                    print(self.name, ": episode: ", episode_count, "mean reward: ", mean_reward)
                if self.number == 0:
                    sess.run(self.increment)
                episode_count += 1

                if episode_count == MAX_NUMBER_OF_EPISODES:
                    self.env.__del__()
                    if self.number == 0:
                        coord.request_stop()
                    else:
                        coord.wait_for_stop()
                    print(self.name, ": work finished")