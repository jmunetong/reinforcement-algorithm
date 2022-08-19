import os
import gym
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



class ReinforceWithBaseline(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        """
        The ReinforceWithBaseline class that inherits from tf.keras.Model.

        The forward pass calculates the policy for the agent given a batch of states. During training,
        ReinforceWithBaseLine estimates the value of each state to be used as a baseline to compare the policy's
        performance with.

        :param state_size: number of parameters that define the state. You don't necessarily have to use this, 
                           but it can be used as the input size for your first dense layer.
        :param num_actions: number of actions in an environment
        """
        super(ReinforceWithBaseline, self).__init__()
        self.num_actions = num_actions
        self.state_size = state_size
        #Optimizer hyperparams
        self.learning_rate = 0.001
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=200,
            decay_rate=0.96, staircase=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)

        ## Actor layers
        self.dense1 = tf.keras.layers.Dense(units=32, activation='relu')
        self.actor = tf.keras.layers.Dense(units=num_actions, activation='linear')
        # Critic Layers
        self.hidden_size = 32
        self.hidden_critic = tf.keras.layers.Dense(units=self.hidden_size, activation='relu')
        self.critic = tf.keras.layers.Dense(units=1, activation='linear')


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
        probs = tf.nn.softmax(self.actor(pass_1))
        return probs

    def value_function(self, states):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An [episode_length, state_size] dimensioned array representing the history of states
        of an episode.
        :return: A [episode_length] matrix representing the value of each state.
        """
        pass_1 = self.hidden_critic(states)
        return self.critic(pass_1)

    def loss(self, states, actions, discounted_rewards):
        """
        Computes the loss for the agent. Refer to the lecture slides referenced in the handout to see how this is done.

        :param states: A batch of states of shape (episode_length, state_size)
        :param actions: History of actions taken at each timestep of the episode (represented as an [episode_length] array)
        :param discounted_rewards: Discounted rewards throughout a complete episode (represented as an [episode_length] array)
        :return: loss, a TensorFlow scalar
        """

        # prepare row indices
        row_indices = tf.range(tf.shape(states)[0])
        # zip row indices with column indices
        pa_indices = tf.stack([row_indices, actions], axis=1)
        pa_i = tf.gather_nd(self.call(states), indices = pa_indices.numpy() )
        v = tf.squeeze(self.value_function(states))
        l_actor = - (tf.reduce_sum(tf.math.log(pa_i) * tf.stop_gradient(discounted_rewards - v )))
        l_critic = tf.reduce_sum((discounted_rewards - v)**2)
        return (l_actor + l_critic)

