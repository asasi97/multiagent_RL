import numpy as np
import operator
import matplotlib.pyplot as plt
import json
import time
import pandas

import pickle
#%matplotlib inline


'''
Notes

For gamma = 1,(10x10) the agent 
	-utilizes the shield less times 
	-is much faster to run 
	-reaches goal quicker
	-doesn't matter if we run all the episodes, it eventually finds the correct path
	

for gamma = 0.9, 
	-slower 
	-utilizes the shield all the times
	-only reaches the goal if we allow agent to reach goal in all the trials

Valuing future rewards rather than the nearby rewards?

'''

class GridWorld:
	def __init__(self):
		self.height = 10
		self.width = 10

		self.grid = np.zeros((self.height, self.width)) - 1 # initializing the values to -1?
		# can replace grid to q_table? jk no there is another called q_table
		
		# self.start_location = (4,0)
		self.start_location = (9,3)
		self.current_location = self.start_location
		
		# 5x5 environment
		# self.obstacles = [(1,3),(1,2),(1,1), (3,3),(3,2),(3,1)]
		# self.goals = [(0,3)]

		# 10x10 environment
		self.obstacles = [(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),(1,9),
		(2,5),
		(3,0),(3,1),(3,2),(3,3),(3,5),
		(4,5),(4,7),(4,8),
		(5,8),(5,9),
		(6,6),
		(7,1),(7,2),(7,3),(7,4),(7,6),
		(8,1),(8,2),(8,4),(8,7),
		(9,9)]
		self.goals = [(0,9)]

		self.terminal_states = [self.obstacles, self.goals]
		
		# Set grid rewards for special cells

		for obs in self.obstacles:
			self.grid[obs[0], obs[1]] = -10

		for g in self.goals:
			self.grid[g[0], g[1]] = 30
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

		# Deterministic environment
		# UP
		# This is the correct sequence of code
		if action == 'UP':
			self.current_location = ( self.current_location[0] - 1, self.current_location[1])
			reward = self.get_reward(self.current_location)
		
		# DOWN
		elif action == 'DOWN':
			self.current_location = ( self.current_location[0] + 1, self.current_location[1])
			reward = self.get_reward(self.current_location)
			
		# LEFT
		elif action == 'LEFT':
			self.current_location = ( self.current_location[0], self.current_location[1] - 1)
			reward = self.get_reward(self.current_location)

		# RIGHT
		elif action == 'RIGHT':
			self.current_location = ( self.current_location[0], self.current_location[1] + 1)
			reward = self.get_reward(self.current_location)		
		return reward

class Q_Agent():
	# Intialise
	def __init__(self, environment, epsilon=0.05, alpha=0.7, gamma=1):
		self.environment = environment
		self.q_table = dict() # Store all Q-values in dictionary of dictionaries 
		for x in range(environment.height): # Loop through all possible grid spaces, create sub-dictionary for each
			for y in range(environment.width):
				self.q_table[(x,y)] = {'UP':0, 'DOWN':0, 'LEFT':0, 'RIGHT':0} 

		self.epsilon = epsilon
		self.alpha = alpha
		self.gamma = gamma
		
	def choose_action(self, available_actions, rand_choose = False):
		"""Returns the optimal action from Q-Value table. If multiple optimal actions, chooses random choice.
		Will make an exploratory random action dependent on epsilon."""

		### We need epsilon decay so that it'll stop exploring after a while
		if np.random.uniform(0,1) < self.epsilon or rand_choose:
			action = available_actions[np.random.randint(0, len(available_actions))]
		else:
			q_values_of_state = self.q_table[self.environment.current_location]
			maxValue = max(q_values_of_state.values())

			# choose random action if there are multiple optimal actions -
			# In the beginning you ahve multiple actions with the same Q-value
			action = np.random.choice([k for k, v in q_values_of_state.items() if v == maxValue])
		
		return action
	
	def learn(self, old_state, reward, new_state, action):
		"""Updates the Q-value table using Q-learning"""
		q_values_of_state = self.q_table[new_state]

		# I think this means the max among the 4 q_values for the actions
		max_q_value_in_new_state = max(q_values_of_state.values())
		current_q_value = self.q_table[old_state][action]

		self.q_table[old_state][action] = (1 - self.alpha) * current_q_value + \
		self.alpha * (reward + self.gamma * max_q_value_in_new_state)

##########################################################################################

def play(environment, agent, trials=500, max_steps_per_episode=1000, prev_qtable=False):	

	# 10x10 environment with obstacles
	with open('multiobject_10x10_Json') as json_file:
		data = json.load(json_file)

	# 5x5 environment with obstacles
	# with open('multiobject_5x5_Json') as json_file:
	# 	data = json.load(json_file)

	epsilon_decay = 0.999
	reward_per_episode = [] 

	if prev_qtable == True:
		agent.q_table = pickle.load(open("safe_qtable_mulobs.pickle", "rb"))

	rchd_goal = 0
	rchd_obstacles = 0 

	all_action_used = []
	hit_obstacle_list =[]

	time_trials = []
	highest_reward = -1000
	min_step = 0
	best_path_taken = []
	
	for trial in range(trials):
		start_time = time.time()
		cumulative_reward = 0 
		step = 0
		game_over = False

		all_action_counter_trial = 0

		# next_node = 3 # 5x5 environment
		next_node = 2275 # 10x10 environment

		path_taken = []

		# For a 10x10 envionment the code is faster when we dont have a max_steps per episode condition

		# while game_over != True: 
		while step < max_steps_per_episode and game_over != True: 
			old_state = environment.current_location
			action = agent.choose_action(environment.actions) 

			path_taken.append(old_state)

			################ TL #####################
			# Maybe create a dictionary of nodes to actions?
			# Mayeb that will be faster with less storage space?
			all_actions_start_node = data['nodes'][str(next_node)]['trans'] # TL
			tl_actions = ['LEFT', 'RIGHT', 'UP', 'DOWN']
			allowable_actions = []
			state_allowable_actions = []
			# loop
			for node in all_actions_start_node:
				possible_action = data['nodes'][str(node)]['state']
				l = len(possible_action)
				possible_action = possible_action[l-4:l]

				# in except you can add 'NO ACTION' later on if there is no action to be taken
				try:
					index = possible_action.index(1)
					allowable_actions.append(tl_actions[index])
					state_allowable_actions.append(node)
				except ValueError:
					pass   
			##########################################


			all_action_counter = 0
			# create a list of allowable_actions for each state
			
			while action not in allowable_actions:				
				# keep doing choose_action if action not in allowable action
				# action = agent.choose_action(environment.actions) 
				# For every time it calls choose_action give it a negative reward


				# action = agent.choose_action(environment.actions, rand_choose = True)
				# Doing rand_choose solves the problem by a bit but it is using exploration
				# Indicates that the q_table selection might be inhibiting?
				# But I want to rely on the q_table later right? 

				# just choose action from list of allowable actions? 
				action = np.random.choice(allowable_actions)

				all_action_counter += 1
				all_action_counter_trial += 1			

			next_node = state_allowable_actions[allowable_actions.index(action)]
			reward = environment.make_step(action)

			new_state = environment.current_location

			# if new_state == (0,9): 
			# 	print('\n')
			# 	print next_node
			# 	print allowable_actions
			# 	print ('Current State -', new_state)
					
			# 	print('Final Action - ', action)
			# 	print('State allowable -', state_allowable_actions)

			agent.learn(old_state, reward, new_state, action)				
			cumulative_reward += reward
			step += 1

			if new_state in environment.goals or new_state in environment.obstacles or step == max_steps_per_episode:

				if new_state in environment.goals:
					rchd_goal += 1 

				if new_state in environment.obstacles:
					rchd_obstacles += 1
					hit_obstacle_list.append((old_state,action))

				environment.__init__()
				game_over = True   

		if highest_reward < cumulative_reward:
			highest_reward = cumulative_reward	
			best_path_taken = path_taken
			min_step = step

		reward_per_episode.append(cumulative_reward) 
		agent.epsilon *= epsilon_decay
		time_trials.append(time.time() - start_time)

		# Maybe add a case later to remove all the zeroes
		if all_action_counter_trial != 0:
			all_action_used.append((trial+1,all_action_counter_trial))

		# # Finding the best path
		# if best_path_taken ==[]:
		# 	best_path_taken = path_taken
		# elif len(path_taken) < len(best_path_taken):
		# 	best_path_taken = path_taken

	#### Create the grid #####################
	grid_visualize = np.zeros((environment.height, environment.width), dtype = object)

	for obs in environment.obstacles:
		grid_visualize[obs[0], obs[1]] = 'X'

	for g in environment.goals:
		grid_visualize[g[0], g[1]] = 'G'

	for path in best_path_taken:
		grid_visualize[path[0], path[1]] = 1

	s_l = environment.start_location
	grid_visualize[s_l[0], s_l[1]] = 'S'
	grid_visualize[grid_visualize == 0] = ''

	df = pandas.DataFrame(grid_visualize)
	print('Printing the best path from S to G\n')
	print(df)
	print('\n')

	##########################################

	# print('Trials that called Shield - ', all_action_used)
	print('Number of trials that called Temporal Logic - ', len(all_action_used), "out of", trials)
	# print('Trials that used Temporal Logic -', all_action_used[])	
	print('Number of times agent reached goal - ', rchd_goal)
	print('Number of times agent hit an obstacle - ', rchd_obstacles)

	print('Highest Reward - ', highest_reward)
	print('Minimum number of steps - ', min_step)

	# Printing the time
	print('Average time taken to run each trial -', np.mean(time_trials))
	print('Max time taken in trial -',np.max(time_trials))

	return reward_per_episode

####################################
starttime_toruncode = time.time()
environment = GridWorld()
agentQ = Q_Agent(environment)

reward_per_episode = play(environment, agentQ, trials=2000, prev_qtable = False)

end_code = time.time() - starttime_toruncode
print('Total time taken to run the code -', end_code)
print('Reward at the end of the episode - ',reward_per_episode[len(reward_per_episode) - 1])
print('Average Reward - ', np.mean(reward_per_episode))

plt.plot(reward_per_episode)
plt.xlabel('Episode Number')
plt.ylabel('Rewards')
plt.savefig('safe_single_ag.png')
plt.show()


pickle.dump(agentQ.q_table, open("safe_qtable_mulobs.pickle", "wb"))


