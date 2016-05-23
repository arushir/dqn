import numpy as np
import random as random
from collections import deque

from cnn import CNN

# See https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf for model description

class DQN:
  def __init__(self, num_actions, observation_shape, capacity, 
                mini_batch_size, epsilon, gamma, params ={}):
    self.epsilon = epsilon
    self.gamma = gamma
    self.num_actions = num_actions
    self.mini_batch_size = mini_batch_size

    # memory 
    self.memory = deque(maxlen=capacity)

    # initialize network
    self.model = CNN(num_actions, observation_shape, params)
    print "model initialized"

  def select_action(self, observation):
    """
    Selects the next action to take based on the current state and learned Q.

    Args:
      observation: the current state

    """

    if random.random() < self.epsilon: 
      # with epsilon probability select a random action 
      action = np.random.randint(0, self.num_actions)
    else:
      # select the action a which maximizes the Q value
      q_values = self.model.predict(observation)
      action = np.argmax(q_values)
    return action

  def update_state(self, action, observation, new_observation, reward, done):
    """
    Stores the most recent action in the replay memory.

    Args: 
      action: the action taken 
      observation: the state before the action was taken
      new_observation: the state after the action is taken
      reward: the reward from the action
      done: a boolean for when the episode has terminated 

    """
    transition = {'action': action,
                  'observation': observation,
                  'new_observation': new_observation,
                  'reward': reward,
                  'is_done': done}
    self.memory.append(transition)

  def get_random_mini_batch(self):
    """
    Gets a random sample of transitions from the replay memory.

    """
    rand_idxs = random.sample(xrange(len(self.memory)), len(self.memory))
    mini_batch = []
    for idx in rand_idxs:
      mini_batch.append(self.memory[idx])
    return mini_batch

  def train_step(self):
    """
    Updates the model based on the mini batch

    """
    if len(self.memory) > self.mini_batch_size:
      mini_batch = self.get_random_mini_batch()

      Xs = []
      ys = []

      for sample in mini_batch:
        y_j = np.zeros(self.num_actions)
        y_j += sample['reward']

        # for nonterminals, add gamma*max_a(Q(phi_{j+1})) term
        if not sample['is_done']:
          q_values = self.model.predict(sample['new_observation'])
          action = np.max(q_values)
          y_j += self.gamma*action

        observation = sample['observation']
        Xs.append(observation.copy())
        ys.append(np.array(y_j))

      Xs = np.array(Xs)
      ys = np.array(ys)

      self.model.train_step(Xs, ys)















