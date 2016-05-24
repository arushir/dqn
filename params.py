import os
import argparse

DEFAULT_EPISODES = 5
DEFAULT_STEPS = 1024
DEFAULT_ENVIRONMENT = 'CartPole-v0'

DEFAULT_MEMORY_CAPACITY = 256
DEFAULT_EPSILON = 0.3
DEFAULT_GAMMA = 0.4
DEFAULT_MINI_BATCH_SIZE = 32

DEFAULT_LEARNING_RATE = 0.001
DEFAULT_REGULARIZATION = 0.001

def parse_args():
  """
  Parses the command line input.

  """
  parser = argparse.ArgumentParser()
  parser.add_argument('-episodes', default = DEFAULT_EPISODES, help = 'number of episodes', type=int)
  parser.add_argument('-steps', default = DEFAULT_STEPS, help = 'number of steps', type=int)
  parser.add_argument('-env', default = DEFAULT_ENVIRONMENT, help = 'environment name', type=str)

  parser.add_argument('-capacity', default = DEFAULT_MEMORY_CAPACITY, help = 'memory capacity', type=int)
  parser.add_argument('-epsilon', default = DEFAULT_EPSILON, help = 'epsilon value for the exploration rate', type=float)
  parser.add_argument('-gamma', default = DEFAULT_GAMMA, help = 'gamma value for the contribution of the Q function in learning', type=float)
  parser.add_argument('-minibatch_size', default = DEFAULT_MINI_BATCH_SIZE, help = 'mini batch size for training', type=int)

  parser.add_argument('-l', default = DEFAULT_LEARNING_RATE, help = 'learning rate', type=float)
  parser.add_argument('-r', default = DEFAULT_REGULARIZATION, help = 'regularization', type=float)

  args = parser.parse_args()
  agent_params = {'episodes': args.episodes, 'steps': args.steps, 'environment': args.env}
  dqn_params = {'memory_capacity': args.capacity, 'epsilon': args.epsilon, 'gamma': args.gamma, 'mini_batch_size': args.minibatch_size}
  cnn_params = {'lr': args.l, 'reg': args.r }

  return agent_params, dqn_params, cnn_params