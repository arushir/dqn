import gym
from params import parse_args
from dqn import DQN
import numpy as np
from collections import deque

def run_dqn():
  # get command line arguments, defaults set in utils.py
  agent_params, dqn_params, cnn_params = parse_args()

  env = gym.make(agent_params['environment'])
  episodes = agent_params['episodes']
  steps = agent_params['steps']
  num_actions = env.action_space.n
  observation_shape = env.observation_space.shape

  # initialize dqn learning
  dqn = DQN(num_actions, observation_shape, dqn_params, cnn_params)

  env.monitor.start('./outputs/cartpole-experiment-1')
  last_100 = deque(maxlen=100)

  for i_episode in range(episodes):
      observation = env.reset()
      reward_sum = 0

      for t in range(steps):
          env.render()
          #print observation

          # select action based on the model
          action = dqn.select_action(observation)
          # execute actin in emulator
          new_observation, reward, done, _ = env.step(action)
          # update the state 
          dqn.update_state(action, observation, new_observation, reward, done)
          observation = new_observation

          # train the model
          dqn.train_step()

          reward_sum += reward
          if done:
              print "Episode ", i_episode
              print "Finished after {} timesteps".format(t+1)
              print "Reward for this episode: ", reward_sum
              last_100.append(reward_sum)
              print "Average reward for last 100 episodes: ", np.mean(last_100)
              break

  env.monitor.close()

if __name__ == '__main__':
  run_dqn()

