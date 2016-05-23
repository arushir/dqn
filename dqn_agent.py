import gym
from utils import parse_args
from dqn import DQN

EPISODES = 5
STEPS = 100

def run_dqn(environment):
	env = gym.make(environment)
	# get command line arguments
	# TODO: pass command line args for parameter tuning to cnn
	# params = parse_args()

	# TODO: how to get shape of action_space?
	print env.action_space
	print env.observation_space.shape

	# TODO: get values from params
	num_actions = 2
	observation_shape = env.observation_space.shape
	capacity = 4
	epsilon = 0.3
	mini_batch_size = 128
	gamma = 0.4

	dqn = DQN(num_actions, observation_shape, capacity, 
                mini_batch_size, epsilon, gamma)


	#env.monitor.start('/tmp/breakout-experiment-1')
	num_steps = 0

	for i_episode in range(EPISODES):
	    observation = env.reset()

	    for t in range(STEPS):
	        env.render()
	        print "observation: "
	        print observation

	        # select action based on the model
	        action = dqn.select_action(observation)
	        # execute actin in emulator
	        new_observation, reward, done, _ = env.step(action)
	        # update the state 
	        dqn.update_state(action, observation, new_observation, reward, done)
	        observation = new_observation

	        # train the model
	        if num_steps > mini_batch_size:
	        	dqn.train_step(observation)

	        if done:
	            print "Episode finished after {} timesteps".format(t+1)
	            break

	        num_steps += 1

	#env.monitor.close()

if __name__ == '__main__':
	run_dqn('CartPole-v0')