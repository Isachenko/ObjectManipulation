from ac_network import AC_Network
from utils.helper import *
from utils.utils import *
from config import *
from utils.csv_summary import CSVSummary
import time

import scipy.misc


class Worker():
    def __init__(self, game, name, s_size, a_size, trainer, model_path, global_episodes,vf, temperature_cte):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        summary_folder = statistics_path + str(self.number)
        self.summary_writer = tf.summary.FileWriter(summary_folder)
        self.csv_summary = CSVSummary(summary_folder, ["time", "step", "value"])


        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(s_size, a_size, self.name, trainer,vf)
        self.update_local_ops = update_target_graph('global', self.name)

        # The Below code is related to setting up the Doom environment
        # game.set_doom_scenario_path("basic.wad")  # This corresponds to the simple task we will pose our agent
        # game.set_doom_map("map01")
        # game.set_screen_resolution(ScreenResolution.RES_160X120)
        # game.set_screen_format(ScreenFormat.GRAY8)
        # game.set_render_hud(False)
        # game.set_render_crosshair(False)
        # game.set_render_weapon(True)
        # game.set_render_decals(False)
        # game.set_render_particles(False)
        # game.add_available_button(Button.MOVE_LEFT)
        # game.add_available_button(Button.MOVE_RIGHT)
        # game.add_available_button(Button.ATTACK)
        # game.add_available_game_variable(GameVariable.AMMO2)
        # game.add_available_game_variable(GameVariable.POSITION_X)
        # game.add_available_game_variable(GameVariable.POSITION_Y)
        # game.set_episode_timeout(300)
        # game.set_episode_start_time(10)
        # game.set_window_visible(False)
        # game.set_sound_enabled(False)
        # game.set_living_reward(-1)
        # game.set_mode(Mode.PLAYER)
        # game.init()
        self.actions = self.actions = np.identity(a_size, dtype=bool).tolist()
        self.env = game

        self.temperature_cte = temperature_cte

    def train(self, rollout, sess, gamma, bootstrap_value, temperature):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        next_observations = rollout[:, 3]
        values = rollout[:, 5]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.inputs: np.vstack(observations),
                     self.local_AC.actions: actions,
                     self.local_AC.advantages: advantages,
                     self.local_AC.state_in[0]: self.batch_rnn_state[0],
                     self.local_AC.state_in[1]: self.batch_rnn_state[1],
                     self.local_AC.temperature: temperature}
        v_l, p_l, e_l, g_n, v_n, self.batch_rnn_state, _ = sess.run([self.local_AC.value_loss,
                                                                     self.local_AC.policy_loss,
                                                                     self.local_AC.entropy,
                                                                     self.local_AC.grad_norms,
                                                                     self.local_AC.var_norms,
                                                                     self.local_AC.state_out,
                                                                     self.local_AC.apply_grads],
                                                                    feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def work(self, max_episode_length, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
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
                temperature  = np.ones(1) * (np.exp(-(episode_count / 300)) + 0.05)
                d = False

                self.env.new_episode()
                if RANDOM == True:
                    self.env.set_target_position_random_X()

                s = self.env.get_state().image
                s = process_frame(s)


                rnn_state = self.local_AC.state_init
                self.batch_rnn_state = rnn_state
                while self.env.is_episode_finished() == False:
                    # Take an action using probabilities from policy network output.
                    a_dist, v, rnn_state = sess.run(
                        [self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
                        feed_dict={self.local_AC.inputs: [s],
                                   self.local_AC.state_in[0]: rnn_state[0],
                                   self.local_AC.state_in[1]: rnn_state[1],
                                   self.local_AC.temperature: temperature})
                    a = np.random.choice(a_dist[0], p=a_dist[0])
                    a = np.argmax(a_dist == a)
                    self.env.make_action(self.actions[a])
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
                        v1 = sess.run(self.local_AC.value,
                                      feed_dict={self.local_AC.inputs: [s],
                                                 self.local_AC.state_in[0]: rnn_state[0],
                                                 self.local_AC.state_in[1]: rnn_state[1],
                                                 self.local_AC.temperature: temperature})[0, 0]
                        v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, v1, temperature)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if d == True:
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the episode buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, 0.0,temperature)

                # Periodically save gifs of episodes, model parameters, and summary statistics.
                #print("Episode count:",episode_count)
                if episode_count % STATISTICS_SAVE_TIME_STEP == 0:
                    if self.number == 0 and episode_count % IMAGE_SAVE_TIME_STEP == 0:
                        time_per_step = 0.05
                        images = np.array(episode_frames)
                        make_gif(images, frames_path + '/image' + str(episode_count) + '.gif',
                                 duration=len(images) * time_per_step, true_image=True, salience=False)

                        #scipy.misc.toimage(images[10], cmin=0.0, cmax=...).save(frames_path + '/image' + str(episode_count) + '.jpg')
                        #scipy.misc.imsave(frames_path + '/image' + str(episode_count) + '.jpg', images[10])

                    #if episode_count % MODEL_SAVE_TIME_STEP == 0 and self.number == 0:
                    #    saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                    #    print("Saved Model")


                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    self.csv_summary.write("Reward_worker_"+ str(self.number)+'_'+EXPERIMENT, [time.time(),episode_count,mean_reward])
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
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