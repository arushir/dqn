import gym
from params import parse_args
from dqn import DQN

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

	env.monitor.start('/tmp/cartpole-experiment-1')
	num_steps = 0

	for i_episode in range(episodes):
	    observation = env.reset()

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

	        if done:
	            print "Episode finished after {} timesteps".format(t+1)
	            break

	        num_steps += 1

	env.monitor.close()

if __name__ == '__main__':
	run_dqn()

