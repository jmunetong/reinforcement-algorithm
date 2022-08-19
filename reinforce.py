import os
import gym
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



class Reinforce(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        """
        The Reinforce class that inherits from tf.keras.Model
        The forward pass calculates the policy for the agent given a batch of states.

        :param state_size: number of parameters that define the state. You don't necessarily have to use this, 
                           but it can be used as the input size for your first dense layer.
        :param num_actions: number of actions in an environment
        """
        super(Reinforce, self).__init__()
        self.learning_rate = 0.001
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=100,
            decay_rate=0.96, staircase=True)

        self.num_actions = num_actions
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        self.dense1 = tf.keras.layers.Dense(units = 32, activation = 'relu')
        self.policy = tf.keras.layers.Dense(units = num_actions, activation = 'linear')

    def call(self, states):
        """
        Performs the forward pass on a batch of states to generate the action probabilities.
        This returns a policy tensor of shape [episode_length, num_actions], where each row is a
        probability distribution over actions for each state.

        :param states: An [episode_length, state_size] dimensioned array
        representing the history of states of an episode
        :return: A [episode_length,num_actions] matrix representing the probability distribution over actions
        for each state in the episode
        """
        pass_1 = self.dense1(states)
        probs = tf.nn.softmax(self.policy(pass_1))
        return probs

    def loss(self, states, actions, discounted_rewards):
        """
        Computes the loss for the agent. Make sure to understand the handout clearly when implementing this.

        :param states: A batch of states of shape [episode_length, state_size]
        :param actions: History of actions taken at each timestep of the episode (represented as an [episode_length] array)
        :param discounted_rewards: Discounted rewards throughout a complete episode (represented as an [episode_length] array)
        :return: loss, a Tensorflow scalar
        """

        # prepare row indices
        row_indices = tf.range(tf.shape(states)[0])
        # zip row indices with column indices
        pa_indices = tf.stack([row_indices, actions], axis=1)
        ## Actions passed in simply
        pa_i = tf.gather_nd(self.call(states), indices = pa_indices.numpy() )
        return - tf.reduce_sum(tf.math.log(pa_i) * discounted_rewards)

