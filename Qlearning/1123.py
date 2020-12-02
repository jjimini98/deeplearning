import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

tf.reset_default_graph()

num_episode = 1000
discount = .99
e = 0.1
learning_rate = 0.1
input_size = not env.observation_space
output_size  = not env.action_space