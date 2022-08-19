import os
import sys
import gym
from pylab import *
import numpy as np
import tensorflow as tf
from reinforce import Reinforce
from reinforce_with_baseline import ReinforceWithBaseline


def visualize_data(total_rewards):
    """
    HELPER - do not edit.
    Takes in array of rewards from each episode, visualizes reward over episodes

    :param total_rewards: List of rewards from all episodes
    """

    x_values = arange(0, len(total_rewards), 1)
    y_values = total_rewards
    plot(x_values, y_values)
    xlabel('episodes')
    ylabel('cumulative rewards')
    title('Reward by Episode')
    grid(True)
    show()


def discount(rewards, discount_factor=.99):
    """
    Takes in a list of rewards for each timestep in an episode, and
    returns a list of the discounted rewards for each timestep.
    Refer to the slides to see how this is done.

    :param rewards: List of rewards from an episode [r_{t1},r_{t2},...]
    :param discount_factor: Gamma discounting factor to use, defaults to .99
    :returns: discounted_rewards: list containing the discounted rewards for each timestep in the original rewards list
    """
    # return [x * discount_factor for x in rewards]
    # return list(map(lambda x: x * discount_factor, rewards))
    prev = 0
    discounted_rewards = np.copy(rewards).astype(np.float32)
    for i in range(1, len(discounted_rewards) + 1):
        discounted_rewards[-i] += prev * discount_factor
        prev = discounted_rewards[-i]
    return discounted_rewards


def generate_trajectory(env, model):
    """
    Generates lists of states, actions, and rewards for one complete episode.

    :param env: The openai gym environment
    :param model: The model used to generate the actions
    :returns: A tuple of lists (states, actions, rewards), where each list has length equal to the number of timesteps in the episode
    """
    states = []
    actions = []
    rewards = []
    state = env.reset()
    done = False

    while not done:
        # 1) use model to generate probability distribution over next actions

        state_expanded = tf.expand_dims(state, axis=0)
        probs = model.call(state_expanded)
        # 2) sample from this distribution to pick the next action
        action = np.random.choice(list(range(probs.numpy().shape[1])), p = np.squeeze(probs))
        states.append(state)
        actions.append(action)
        state, rwd, done, _ = env.step(action)
        rewards.append(rwd)

    return states, actions, rewards


def train(env, model):
    """
    This function should train your model for one episode.
    Each call to this function should generate a complete trajectory for one
    episode (lists of states, action_probs, and rewards seen/taken in the episode), and
    then train on that data to minimize your model loss.
    Make sure to return the total reward for the episode

    :param env: The openai gym environment
    :param model: The model
    :returns: The total reward for the episode
    """

    with tf.GradientTape() as tape:
        states, actions, rewards = generate_trajectory(env, model)
        discount_rewards = discount(rewards)
        loss=  model.loss(np.asarray(states), actions, discount_rewards)
        print("Loss value: {}".format(loss.numpy()))

    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return sum(rewards)

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"REINFORCE", "REINFORCE_BASELINE"}:
        print("USAGE: python assignment.py <Model Type>")
        print("<Model Type>: [REINFORCE/REINFORCE_BASELINE]")
        exit()

    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    num_actions = env.action_space.n

    # Initialize model
    if sys.argv[1] == "REINFORCE":
        model = Reinforce(state_size, num_actions) 
    elif sys.argv[1] == "REINFORCE_BASELINE":
        model = ReinforceWithBaseline(state_size, num_actions)

    EPOCHS = 650
    rewards = []
    for e in range (EPOCHS):
        print(e)
        rewards.append(train(env, model))
    print(mean(rewards[-50:]))
    visualize_data(rewards)

if __name__ == '__main__':
    main()

