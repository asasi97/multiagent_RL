import numpy as np
import operator
import matplotlib.pyplot as plt

import pandas

import pickle
import time

from ppo_grid import PPO
from ppo_grid import ActorCritic

import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym

from torch.autograd import Variable

#%matplotlib inline

# THE INIT FILE HAS TO BE OUTSIDE INCASE THE CODE USES THE ENTIRE ITERATION STEPS
# CHANGE IN ALL THE CODES


# LIMITATION - Cant go back because each state has a fixed action

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class GridWorld:
	def __init__(self):

		# 5x5
		self.height = 5
		self.width = 5

		self.grid = np.zeros((self.height, self.width)) - 1 # initializing the values to -1?
		# can replace grid to q_table? jk no there is another called q_table
		
		# Set random start location for the agent
		#self.current_location = ( 4, np.random.randint(0,5))

		self.start_location = (4,0)
		self.current_location = self.start_location
		
		# 5x5 environment
		self.obstacles = [(1,3)] #,(1,2),(1,1),(3,3),(3,2),(3,1)]
		# self.goals = [(0,3),(4,4),(2,2)]
		# self.goals = [(0, 3), (2, 4), (4, 2)]
		self.goals = [(0, 3)]

		self.terminal_states = [self.obstacles, self.goals]
		
		# Set grid rewards for special cells

		for obs in self.obstacles:
			self.grid[obs[0], obs[1]] = -10

		# for g in self.goals:
		# 	self.grid[g[0], g[1]] = 2
		#  grid[a][b] = grid[a,b]

		self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']


	def get_available_actions(self):
		return self.actions
	
	def agent_on_map(self):
		"""Prints out current location of the agent on the grid (used for debugging)"""
		grid = np.zeros(( self.height, self.width))
		grid[ self.current_location[0], self.current_location[1]] = 1
		return grid
	
	def get_reward(self, new_location):
		return self.grid[new_location[0], new_location[1]]
		
	
	def make_step(self, action):
		"""Moves the agent in the specified direction. If agent is at a border, agent stays still
		but takes negative reward. Function returns the reward for the move."""
		# Store previous location
		
		last_location = self.current_location
		# UP
		if action == 'UP':
			# If agent is at the top, stay still, collect reward
			if last_location[0] == 0:
				reward = self.get_reward(last_location)
			else:
				self.current_location = ( self.current_location[0] - 1, self.current_location[1])
				reward = self.get_reward(self.current_location)
		
		# DOWN
		elif action == 'DOWN':
			# If agent is at bottom, stay still, collect reward
			if last_location[0] == self.height - 1:
				reward = self.get_reward(last_location)
			else:
				self.current_location = ( self.current_location[0] + 1, self.current_location[1])
				reward = self.get_reward(self.current_location)
			
		# LEFT
		elif action == 'LEFT':
			# If agent is at the left, stay still, collect reward
			if last_location[1] == 0:
				reward = self.get_reward(last_location)
			else:
				self.current_location = ( self.current_location[0], self.current_location[1] - 1)
				reward = self.get_reward(self.current_location)

		# RIGHT
		elif action == 'RIGHT':
			# If agent is at the right, stay still, collect reward
			if last_location[1] == self.width - 1:
				reward = self.get_reward(last_location)
			else:
				self.current_location = ( self.current_location[0], self.current_location[1] + 1)
				reward = self.get_reward(self.current_location)
				
		return reward

####################################
starttime_toruncode = time.time()

# env_name = "MountainCar-v0"
# # creating environment
# env = gym.make(env_name)
state_dim = 2
action_dim = 4
print(state_dim)
print(action_dim)
render = False
solved_reward = 230         # stop training if avg_reward > solved_reward
log_interval = 100           # print avg reward in the interval
max_episodes = 5000        # max training episodes
max_timesteps = 1000         # max timesteps in one episode
n_latent_var = 64           # number of variables in hidden layer
update_timestep = 400      # update policy every n timesteps
lr = 0.002
betas = (0.9, 0.999)
gamma = 0.99                # discount factor
K_epochs = 4                # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
random_seed = None
#############################################
	
if random_seed:
	torch.manual_seed(random_seed)
	env.seed(random_seed)

memory = Memory()
ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
environment = GridWorld()

# logging variables
running_reward = 0
avg_length = 0
timestep = 0

rchd_obstacles = 0 
time_trials = []
rchd_order = 0
highest_reward = -1000
best_path_taken = []
reward_per_episode = [] 

for i_episode in range(1, max_episodes+1):
	start_time = time.time()
	cumulative_reward = 0 
	step = 0
	game_over = False
	wrong_order = False
	reached_allgoals = False
	encountered_goals = []

	path_taken = []

	environment.__init__()

	while step < max_timesteps and game_over != True:
		timestep += 1 
		old_state = environment.current_location
		path_taken.append(old_state)

		state = np.array(old_state)
		state = Variable(torch.from_numpy(state))
		print(state)
		print('State -', state)
		action = ppo.policy_old.act(state, memory)

		reward = environment.make_step(action)
		new_state = environment.current_location
		memory.rewards.append(reward)
		# memory.is_terminals.append(done)

		if new_state in environment.goals and new_state not in encountered_goals:
			encountered_goals.append(new_state)				
			if environment.goals == encountered_goals:
				reward = 20
				reached_allgoals = True
				rchd_order += 1 
				best_path_taken = path_taken
			else:
				# something wrong with this code maybe?
				if new_state != environment.goals[len(encountered_goals) - 1]:
					reward = -10
					wrong_order = True
				else:
					reward = 10
		
		if timestep % update_timestep == 0:
			ppo.update(memory)
			memory.clear_memory()
			timestep = 0
				
		cumulative_reward += reward
		running_reward += reward
		step += 1

		length_match = (len(encountered_goals) == len(environment.goals))
		if new_state in environment.obstacles or wrong_order or reached_allgoals or length_match:
			if new_state in environment.obstacles:
				rchd_obstacles += 1
			game_over = True 

	
	if highest_reward < cumulative_reward:
		highest_reward = cumulative_reward
		# best_path_taken = path_taken

	reward_per_episode.append(cumulative_reward) 
	time_trials.append(time.time() - start_time)

	avg_length += t
	
	# stop training if avg_reward > solved_reward
	if running_reward > (log_interval*solved_reward):
		print("########## Solved! ##########")
		torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format(env_name))
		break
		
	# logging
	if i_episode % log_interval == 0:
		avg_length = int(avg_length/log_interval)
		running_reward = int((running_reward/log_interval))
		
		print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
		running_reward = 0
		avg_length = 0

#### Create the grid #####################
grid_visualize = np.zeros((environment.height, environment.width), dtype = object)

print('Best path taken -',best_path_taken)
for path in best_path_taken:
	grid_visualize[path[0], path[1]] = 1

for obs in environment.obstacles:
	grid_visualize[obs[0], obs[1]] = 'X'

g_count = 1
for g in environment.goals:
	grid_visualize[g[0], g[1]] = 'G' + str(g_count)
	g_count += 1

s_l = environment.start_location
grid_visualize[s_l[0], s_l[1]] = 'S'
grid_visualize[grid_visualize == 0] = ''

df = pandas.DataFrame(grid_visualize)
print(df)


end_code = time.time() - starttime_toruncode
print('Total time taken to run the code -', end_code)
print('Reward at the end of the episode - ',reward_per_episode[len(reward_per_episode) - 1])
print('Average Reward - ', np.mean(reward_per_episode))

plt.plot(reward_per_episode)
plt.xlabel('Episode Number')
plt.ylabel('Rewards')
plt.savefig('single_ag_grid.png')
plt.show()


# pickle.dump(agentQ.q_table, open("qtable_saved.pickle", "wb"))