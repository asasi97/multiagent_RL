import numpy as np
import operator
import matplotlib.pyplot as plt

import pandas

import pickle
import time

import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

# from IPython.display import clear_output
import matplotlib.pyplot as plt

# Use CUDA
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


# THE INIT FILE HAS TO BE OUTSIDE INCASE THE CODE USES THE ENTIRE ITERATION STEPS
# CHANGE IN ALL THE CODES


# LIMITATION - Cant go back because each state has a fixed action

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
		self.obstacles = [(1,3),(1,2),(1,1),(3,3),(3,2),(3,1)]
		# self.goals = [(0,3),(4,4),(2,2)]
		self.goals = [(0, 3), (2, 4), (4, 2)]

		self.terminal_states = [self.obstacles, self.goals]
		
		# Set grid rewards for special cells

		for obs in self.obstacles:
			self.grid[obs[0], obs[1]] = -10

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


# Neural Network
def init_weights(m):
	if isinstance(m, nn.Linear):
		nn.init.normal_(m.weight, mean=0., std=0.1)
		nn.init.constant_(m.bias, 0.1)
		

class ActorCritic(nn.Module):
	def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
		super(ActorCritic, self).__init__()
		
		self.critic = nn.Sequential(
			nn.Linear(num_inputs, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, 1)
		)
		
		self.actor = nn.Sequential(
			nn.Linear(num_inputs, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, num_outputs),
		)
		self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
		
		self.apply(init_weights)
		
	def forward(self, x):
		value = self.critic(x)
		mu    = self.actor(x)
		std   = self.log_std.exp().expand_as(mu)
		dist  = Normal(mu, std)
		return dist, value


def plot(frame_idx, rewards):
	# clear_output(True)
	plt.figure(figsize=(20,5))
	plt.subplot(131)
	plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
	plt.plot(rewards)
	plt.show()
	
def test_env(vis=False):
	state = env.reset()
	if vis: env.render()
	done = False
	total_reward = 0
	while not done:
		state = torch.FloatTensor(state).unsqueeze(0).to(device)
		dist, _ = model(state)
		next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
		state = next_state
		if vis: env.render()
		total_reward += reward
	return total_reward

# GAE
def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
	values = values + [next_value]
	gae = 0
	returns = []
	for step in reversed(range(len(rewards))):
		delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
		gae = delta + gamma * tau * masks[step] * gae
		returns.insert(0, gae + values[step])
	return returns 


# PPO
def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
	batch_size = states.size(0)
	for _ in range(batch_size // mini_batch_size):
		rand_ids = np.random.randint(0, batch_size, mini_batch_size)
		yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
		
		

def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
	for _ in range(ppo_epochs):
		for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
			dist, value = model(state)
			entropy = dist.entropy().mean()
			new_log_probs = dist.log_prob(action)

			ratio = (new_log_probs - old_log_probs).exp()
			surr1 = ratio * advantage
			surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

			actor_loss  = - torch.min(surr1, surr2).mean()
			critic_loss = (return_ - value).pow(2).mean()

			loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
##########################################
starttime_toruncode = time.time()
environment = GridWorld()

num_inputs  = 2
num_outputs = 4

#Hyper params:
hidden_size      = 256
lr               = 3e-4
num_steps        = 20
mini_batch_size  = 5
ppo_epochs       = 4
threshold_reward = -200
trials=10000
max_steps_per_episode=1000

model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

max_frames = 15000
frame_idx  = 0
test_rewards = []

environment.__init__()
early_stop = False

epsilon_decay = 0.999
reward_per_episode = [] 

# if prev_qtable == True:
# 	agent.q_table = pickle.load(open("qtable_saved.pickle", "rb"))

rchd_obstacles = 0 
time_trials = []
rchd_order = 0

highest_reward = -1000
best_path_taken = []

for trial in range(trials):
	start_time = time.time()
	cumulative_reward = 0 
	step = 0
	game_over = False
	wrong_order = False
	reached_allgoals = False
	encountered_goals = []

	path_taken = []

	log_probs = []
	values    = []
	states    = []
	actions   = []
	rewards   = []
	masks     = []
	entropy = 0

	while step < max_steps_per_episode and game_over != True: 
		old_state = environment.current_location
		path_taken.append(old_state)

		state = torch.FloatTensor(old_state).to(device)
		print('State',state)
		dist, value = model(state)

		action = dist.sample()

		reward = environment.make_step(action)
		new_state = environment.current_location

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

		log_prob = dist.log_prob(action)
		entropy += dist.entropy().mean()
		log_probs.append(log_prob)
		values.append(value)
		rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
		masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
		
		states.append(state)
		actions.append(action)
		
		state = next_state
		frame_idx += 1

		if frame_idx % 1000 == 0:
			test_reward = np.mean([test_env() for _ in range(10)])
			test_rewards.append(test_reward)
			plot(frame_idx, test_rewards)
			if test_reward > threshold_reward: early_stop = True
			
		cumulative_reward += reward

		step += 1

		length_match = (len(encountered_goals) == len(environment.goals))
		if new_state in environment.obstacles or wrong_order or reached_allgoals or length_match:
			if new_state in environment.obstacles:
				rchd_obstacles += 1
			game_over = True  

	environment.__init__()
	if highest_reward < cumulative_reward:
		highest_reward = cumulative_reward
		# best_path_taken = path_taken

	reward_per_episode.append(cumulative_reward) 
	
	time_trials.append(time.time() - start_time)
	next_state = torch.FloatTensor(next_state).to(device)
	_, next_value = model(next_state)
	returns = compute_gae(next_value, rewards, masks, values)

	returns   = torch.cat(returns).detach()
	log_probs = torch.cat(log_probs).detach()
	values    = torch.cat(values).detach()
	states    = torch.cat(states)
	actions   = torch.cat(actions)
	advantage = returns - values
	
	ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)


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

##########################################

print('Number of times agent reached goals in specific order- ', rchd_order)
print('Number of times agent hit an obstacle - ', rchd_obstacles)
print('Highest Reward -', highest_reward)

# Printing the time
print('Average time taken to run each trial -', np.mean(time_trials))
print('Max time taken in trial -',np.max(time_trials))


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