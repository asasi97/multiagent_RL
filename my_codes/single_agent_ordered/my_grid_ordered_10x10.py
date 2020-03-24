import numpy as np
import operator
import matplotlib.pyplot as plt

import pandas

import pickle
import time

#%matplotlib inline

# We need negative rewards for the steps to encourage the code to find the shortest path from start to goal

class GridWorld:
	def __init__(self):


		# 10x10
		self.height = 10
		self.width = 10

		self.grid = np.zeros((self.height, self.width)) - 1 # initializing the values to -1?
		# can replace grid to q_table? jk no there is another called q_table
		
		# Set random start location for the agent
		#self.current_location = ( 4, np.random.randint(0,5))

		self.start_location = (9,3)
		self.current_location = self.start_location


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
		self.goals = [(6,7),(0,9)]

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

class Q_Agent():
	# Intialise
	def __init__(self, environment, epsilon=0.1, alpha=0.1, gamma=0.9):
		self.environment = environment
		self.q_table = dict() # Store all Q-values in dictionary of dictionaries 
		for x in range(environment.height): # Loop through all possible grid spaces, create sub-dictionary for each
			for y in range(environment.width):
				self.q_table[(x,y)] = {'UP':0, 'DOWN':0, 'LEFT':0, 'RIGHT':0} 

		self.epsilon = epsilon
		self.alpha = alpha
		self.gamma = gamma
		
	def choose_action(self, available_actions):
		"""Returns the optimal action from Q-Value table. If multiple optimal actions, chooses random choice.
		Will make an exploratory random action dependent on epsilon."""
		### We need epsilon decay so that it'll stop exploring after a while
		if np.random.uniform(0,1) < self.epsilon:
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

def play(environment, agent, trials=500, max_steps_per_episode=1000, prev_qtable=False):

	epsilon_decay = 0.9998
	reward_per_episode = [] 

	if prev_qtable == True:
		agent.q_table = pickle.load(open("qtable_saved.pickle", "rb"))

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

		while step < max_steps_per_episode and game_over != True: 
			old_state = environment.current_location
			path_taken.append(old_state)

			action = agent.choose_action(environment.actions) 
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

			# Not my code
			# if not encountered_goals:
			# 	if new_state == environment.goals[0]:
			# 		encountered_goals.append(new_state)
			# 		reward = 10
			# 	elif new_state == environment.goals[1]:
			# 		wrong_order = True
			# 		reward = -10
			# if encountered_goals == [environment.goals[0]]:
			# 	if new_state == environment.goals[1]:
			# 		reward = 20
			# 		reached_allgoals = True
			# 		rchd_order += 1
			

			agent.learn(old_state, reward, new_state, action)				
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
			best_path_taken = path_taken

		reward_per_episode.append(cumulative_reward) 
		agent.epsilon *= epsilon_decay
		time_trials.append(time.time() - start_time)
	
	#### Create the grid #####################
	grid_visualize = np.zeros((environment.height, environment.width), dtype = object)

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

	return reward_per_episode

####################################
starttime_toruncode = time.time()
environment = GridWorld()
agentQ = Q_Agent(environment)

trials = 20000
reward_per_episode = play(environment, agentQ, trials=20000, prev_qtable = False)

end_code = time.time() - starttime_toruncode
print('Total time taken to run the code -', end_code)
print('Reward at the end of the episode - ',reward_per_episode[len(reward_per_episode) - 1])
print('Average Reward - ', np.mean(reward_per_episode))

plt.plot(reward_per_episode)
# plt.scatter(range(trials),reward_per_episode)
plt.xlabel('Episode Number')
plt.ylabel('Rewards')
plt.savefig('single_ag_grid.png')
plt.show()


pickle.dump(agentQ.q_table, open("qtable_saved.pickle", "wb"))