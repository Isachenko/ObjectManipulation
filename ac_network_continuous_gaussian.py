import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import utils


ENTROPY_BETA = 0.01
A_BOUND_LOW = -1
A_BOUND_HIGH = 1

class ACNetworkContinuousGaussian():
    def __init__(self, s_size, a_size, scope, trainer):


        with tf.variable_scope(scope):
            # Input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
            self.imageIn = tf.reshape(self.inputs, shape=[-1, 84, 84, 1])
            self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=self.imageIn, num_outputs=16,
                                     kernel_size=[8, 8], stride=[4, 4], padding='VALID')
            self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=self.conv1, num_outputs=32,
                                     kernel_size=[4, 4], stride=[2, 2], padding='VALID')
            hidden = slim.fully_connected(slim.flatten(self.conv2), 256, activation_fn=tf.nn.elu)

            # Recurrent network for temporal dependencies
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(hidden, [0])
            step_size = tf.shape(self.imageIn)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])

            # Output layers for policy and value estimations
            self.policy_mean = slim.fully_connected(rnn_out, a_size,
                                               activation_fn=tf.nn.tanh,
                                               weights_initializer=utils.normalized_columns_initializer(0.01),
                                               biases_initializer=None)
            self.policy_sigma = slim.fully_connected(rnn_out, a_size,
                                               activation_fn=tf.nn.softplus,
                                               weights_initializer=utils.normalized_columns_initializer(0.01),
                                               biases_initializer=None)
            self.value = slim.fully_connected(rnn_out, 1,
                                              activation_fn=None,
                                              weights_initializer=utils.normalized_columns_initializer(1.0),
                                              biases_initializer=None)


            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None, a_size], dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)
                self.print_adv = tf.Print(self.advantages, [self.advantages])
                self.actions_reshaped = tf.reshape(self.actions, shape=[-1, a_size])
                #self.actions_diff = tf.reduce_sum(tf.square(self.actions_reshaped - self.policy), [1]) #think more


                mu = self.policy_mean * A_BOUND_HIGH
                sigma = self.policy_sigma + 1e-4

                normal_dist = tf.contrib.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
                self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), A_BOUND_LOW, A_BOUND_HIGH)

                td = tf.subtract(self.target_v, tf.reshape(self.value, [-1]), name='TD_error')
                print("td: ", td)
                #td = self.target_v - self.value
                #value loss
                self.value_loss = tf.reduce_mean(tf.square(td))

                #action loss
                log_prob = normal_dist.log_prob(self.actions, name='log_prob')
                #reduced_log_prob = tf.reduce_sum(log_prob, 1)
                self.exp_v = tf.multiply(log_prob, td, name="mult_log_td")
                self.entropy = normal_dist.entropy()  # encourage exploration
                #self.print_entropy = tf.Print(self.entropy, [self.entropy])
                self.exp_v = ENTROPY_BETA * self.entropy + self.exp_v
                #self.print_exp_v = tf.Print(self.exp_v, [self.exp_v])
                self.policy_loss = tf.reduce_mean(-self.exp_v) #+ self.print_exp_v + self.print_entropy
                #self.print_policy_loss = tf.Print(self.policy_loss, [self.policy_loss])
                self.mean_entropy = tf.reduce_mean(self.entropy)

                self.loss = 0.5 * self.value_loss + self.policy_loss #+ self.print_policy_loss

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

                # Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))