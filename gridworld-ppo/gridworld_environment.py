import numpy as np

class GridWorld:
	def __init__(self):

		# 5x5
		self.observation_space = 2
		self.action_space = 4	

		# # 5x5 environment
		self.height = 5
		self.width = 5
		self.start_location = (4,0)
		self.current_location = self.start_location
		self.obstacles = [(1,3),(1,2),(1,1),(3,3),(3,2),(3,1)]
		self.goals = [(0,3),(4,4),(2,2)]
		# self.goals = [(0, 3), (2, 4), (4, 2)]
        # self.goals = [(0,3)]

		# 10x10 environment
		# self.height = 10
		# self.width = 10
		# self.start_location = (9,3)
		# self.current_location = self.start_location
		# self.obstacles = [(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),(1,9),
		# (2,5),
		# (3,0),(3,1),(3,2),(3,3),(3,5),
		# (4,5),(4,7),(4,8),
		# (5,8),(5,9),
		# (6,6),
		# (7,1),(7,2),(7,3),(7,4),(7,6),
		# (8,1),(8,2),(8,4),(8,7),
		# (9,9)]
		# # self.goals = [(6,7),(0,9)]
		# self.goals = [(0,9)]

		self.grid = np.zeros((self.height, self.width)) - 1
		self.terminal_states = [self.obstacles, self.goals]

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
#         0 - UP
#         1 - DOWN
#         2 - LEFT
#         3 - RIGHT

		last_location = self.current_location
		reward_ret = 0
		# UP
		if action == 0:
			# If agent is at the top, stay still, collect reward
			if last_location[0] == 0:
				reward_ret = self.get_reward(last_location)
			else:
				self.current_location = ( self.current_location[0] - 1, self.current_location[1])
				reward_ret = self.get_reward(self.current_location)

		# DOWN
		elif action == 1:
			# If agent is at bottom, stay still, collect reward
			if last_location[0] == self.height - 1:
				reward_ret = self.get_reward(last_location)
			else:
				self.current_location = ( self.current_location[0] + 1, self.current_location[1])
				reward_ret = self.get_reward(self.current_location)

		# LEFT
		elif action == 2:
			# If agent is at the left, stay still, collect reward
			if last_location[1] == 0:
				reward_ret = self.get_reward(last_location)
			else:
				self.current_location = ( self.current_location[0], self.current_location[1] - 1)
				reward_ret = self.get_reward(self.current_location)

		# RIGHT
		elif action == 3:
			# If agent is at the right, stay still, collect reward
			if last_location[1] == self.width - 1:
				reward_ret = self.get_reward(last_location)
			else:
				self.current_location = ( self.current_location[0], self.current_location[1] + 1)
				reward_ret = self.get_reward(self.current_location)

		return reward_ret