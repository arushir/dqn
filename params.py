import os
import argparse

DEFAULT_EPISODES = 2000
DEFAULT_STEPS = 500
DEFAULT_ENVIRONMENT = 'CartPole-v0'

DEFAULT_MEMORY_CAPACITY = 10000
DEFAULT_EPSILON = 0.1
DEFAULT_GAMMA = 0.9
DEFAULT_MINI_BATCH_SIZE = 10

DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_REGULARIZATION = 0.01
DEFAULT_NUM_HIDDEN = 2
DEFAULT_HIDDEN_SIZE = 20

def parse_args():
  """
  Parses the command line input.

  """
  parser = argparse.ArgumentParser()
  parser.add_argument('-episodes', default = DEFAULT_EPISODES, help = 'number of episodes', type=int)
  parser.add_argument('-steps', default = DEFAULT_STEPS, help = 'number of steps', type=int)
  parser.add_argument('-env', default = DEFAULT_ENVIRONMENT, help = 'environment name', type=str)

  parser.add_argument('-capacity', default = DEFAULT_MEMORY_CAPACITY, help = 'memory capacity', type=int)
  parser.add_argument('-epsilon', default = DEFAULT_EPSILON, help = 'epsilon value for the probability of taking a random action', type=float)
  parser.add_argument('-gamma', default = DEFAULT_GAMMA, help = 'gamma value for the contribution of the Q function in learning', type=float)
  parser.add_argument('-minibatch_size', default = DEFAULT_MINI_BATCH_SIZE, help = 'mini batch size for training', type=int)

  parser.add_argument('-l', default = DEFAULT_LEARNING_RATE, help = 'learning rate', type=float)
  parser.add_argument('-r', default = DEFAULT_REGULARIZATION, help = 'regularization', type=float)
  parser.add_argument('-num_hidden', default = DEFAULT_NUM_HIDDEN, help = 'the number of hidden layers in the deep network', type=int)
  parser.add_argument('-hidden_size', default = DEFAULT_HIDDEN_SIZE, help = 'the hidden size of all layers in the network', type=int)


  args = parser.parse_args()
  agent_params = {'episodes': args.episodes, 'steps': args.steps, 'environment': args.env}
  dqn_params = {'memory_capacity': args.capacity, 'epsilon': args.epsilon, 'gamma': args.gamma, 'mini_batch_size': args.minibatch_size}
  cnn_params = {'lr': args.l, 'reg': args.r, 'num_hidden': args.num_hidden, 'hidden_size': args.hidden_size, 'mini_batch_size': args.minibatch_size}

  return agent_params, dqn_params, cnn_params