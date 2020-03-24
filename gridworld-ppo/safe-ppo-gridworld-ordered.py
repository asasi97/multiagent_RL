# https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py?fbclid=IwAR0iP1F7rT6JhZf-5ttngd_BNhmtugkT602VfewuG_FTyJDgQzphGnUT-HI

import numpy as np
import operator
import matplotlib.pyplot as plt

import pandas
import pickle
import time
import json

from ppo_grid import Memory
from ppo_grid import ActorCritic
from ppo_grid import PPO

from gridworld_environment import GridWorld

starttime_toruncode = time.time()
############## Hyperparameters ##############
environment = GridWorld()
state_dim = 2
action_dim = 4
render = False
solved_reward = 230         # stop training if avg_reward > solved_reward
log_interval = 500           # print avg reward in the interval
max_episodes = 50000        # max training episodes
max_timesteps = 300         # max timesteps in one episode
n_latent_var = 64           # number of variables in hidden layer
update_timestep = 1000      # update policy every n timesteps
lr = 0.002
betas = (0.9, 0.999)
gamma = 1       #0.99         # discount factor
K_epochs = 4                # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
random_seed = None
trials=5000
max_steps_per_episode=1000
#############################################

memory = Memory()
ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)

# with open('multiobject_10x10_Json') as json_file:
# 	data = json.load(json_file)

with open('multiobject_5x5_Json') as json_file:
	data = json.load(json_file)

# logging variables
running_reward = 0
avg_length = 0
timestep = 0

reward_per_episode = []
rchd_obstacles = 0 
time_trials = []
rchd_order = 0
highest_reward = -1000
best_path_taken = []
all_action_used = []

# training loop
for trial in range(trials):
	start_time = time.time()
	cumulative_reward = 0 
	step = 0
	game_over = False
	wrong_order = False
	reached_allgoals = False
	encountered_goals = []
	path_taken = []

	# Safe
	all_action_counter_trial = 0
	next_node = 3 # 5x5 environment
	# next_node = 2275 # 10x10 environment
	done = False
	environment.__init__()
	while step < max_steps_per_episode and game_over != True:
		timestep += 1
		
		state = environment.current_location
		state = np.array(state)
		path_taken.append(state)
		
		# Running policy_old:
		action = ppo.policy_old.act(state, memory, False, '0')

		################ TL #####################
		# Maybe create a dictionary of nodes to actions?
		# Maybe that will be faster with less storage space?
		all_actions_start_node = data['nodes'][str(next_node)]['trans'] # TL
		tl_actions = ['LEFT', 'RIGHT', 'UP', 'DOWN']
		grid_actions = [2,3,0,1]
		allowable_actions = []
		allowable_actions_just_index = [] #
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
				allowable_actions_just_index.append(grid_actions[index]) #
				state_allowable_actions.append(node)
			except ValueError:
				pass   

		act_counter = 0
		while action not in allowable_actions_just_index:
			# print('Got in')
			# action = ppo.policy_old.act(state, memory)

			# action = np.random.choice(allowable_actions)

			# if act_counter == 100:
			# 	action = np.random.choice(allowable_actions)
				#need to put another counter here

			action = np.random.choice(allowable_actions_just_index) #
			all_action_counter_trial += 1
			act_counter += 1		

		# next_node = state_allowable_actions[allowable_actions.index(action)]
		next_node = state_allowable_actions[allowable_actions_just_index.index(action)]	#
		# if action == 'UP':
		# 	action = 0
		# elif action == 'DOWN':
		# 	action = 1
		# elif action == 'LEFT':
		# 	action = 2
		# elif action == 'RIGHT':
		# 	action = 3

		ppo.policy_old.act(state, memory, True, action)
		##########################################

		reward = environment.make_step(action)
		new_state = environment.current_location

		if new_state in environment.goals and new_state not in encountered_goals:
			encountered_goals.append(new_state)
			if environment.goals == encountered_goals:
				# reward = 20
				reward +=200
				reached_allgoals = True
				rchd_order += 1 
				best_path_taken = path_taken
				done = True
			else:
				# something wrong with this code maybe?
				if new_state != environment.goals[len(encountered_goals) - 1]:
					reward = -10 # Is this -10 confusing with the obstacle
					wrong_order = True
				else:
					reward += 100
					# reward = 10
		

		length_match = (len(encountered_goals) == len(environment.goals))
		if new_state in environment.obstacles or wrong_order or reached_allgoals or length_match:
			if new_state in environment.obstacles:
				rchd_obstacles += 1
			game_over = True

		# Saving reward:
		memory.rewards.append(reward)
		memory.is_terminals.append(done)
		
		# update if its time
		if timestep % update_timestep == 0:
			ppo.update(memory)
			memory.clear_memory()
			timestep = 0
		
		running_reward += reward
		cumulative_reward += reward
		step += 1

			
	avg_length += step
	i_episode = trial + 1
	
	# stop training if avg_reward > solved_reward
	# if running_reward > (log_interval*solved_reward):
	# 	print("########## Solved! ##########")
	# 	torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format(env_name))
	# 	break
		
	# logging
	if i_episode % log_interval == 0:
		# avg_length = int(avg_length/log_interval)
		running_reward = int((running_reward/log_interval))
		
		print('Episode {} \t Steps: {} \t reward: {}'.format(i_episode, step, cumulative_reward))
		print('Number of times agent reached goals in specific order- ', rchd_order)
		print
		running_reward = 0
		avg_length = 0
		
	if highest_reward < cumulative_reward:
		highest_reward = cumulative_reward
		best_path_taken = path_taken

	reward_per_episode.append(cumulative_reward) 

	if all_action_counter_trial != 0:
		all_action_used.append((trial+1,all_action_counter_trial))

	time_trials.append(time.time() - start_time)

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
print('Number of trials that called Temporal Logic - ', len(all_action_used), "out of", trials)
print('Number of times agent reached goals in specific order- ', rchd_order)
print('Number of times agent hit an obstacle - ', rchd_obstacles)

print('Min number of steps -', len(best_path_taken))
print('Highest Reward -', highest_reward)

# Printing the time
print('Average time taken to run each trial -', np.mean(time_trials))
print('Max time taken in trial -',np.max(time_trials))

print('Reward at the end of the episode - ',reward_per_episode[len(reward_per_episode) - 1])
print('Average Reward - ', np.mean(reward_per_episode))
# print('Numer of times random TL action was used -',all_action_used)
end_code = time.time() - starttime_toruncode
print('Total time taken to run the code -', end_code)

plt.plot(reward_per_episode)
plt.xlabel('Episode Number')
plt.ylabel('Rewards')
plt.savefig('single_ag_grid.png')
plt.show()

##########




