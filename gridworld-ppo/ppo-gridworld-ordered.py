# https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py?fbclid=IwAR0iP1F7rT6JhZf-5ttngd_BNhmtugkT602VfewuG_FTyJDgQzphGnUT-HI

# Why is it staying on the same cell?
# Is there a wrong action being passed?

import numpy as np
import operator
import matplotlib.pyplot as plt

import pandas
import pickle
import time

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
update_timestep = 2000      # update policy every n timesteps
lr = 0.002
betas = (0.9, 0.999)
gamma = 0.99                # discount factor
K_epochs = 4                # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
random_seed = None
trials=10000
max_steps_per_episode=1000
#############################################

memory = Memory()
ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)

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
	
	environment.__init__()
	done = False
	while step < max_steps_per_episode and game_over != True:
		timestep += 1
		
		state = environment.current_location
		state = np.array(state)
#             print(state)
		path_taken.append(state)
		
		# Running policy_old:
		action = ppo.policy_old.act(state, memory)
#             print(action)
		reward = environment.make_step(action)

		if reward == 0:
			print(state)
			print(reward)

		new_state = environment.current_location

		if new_state in environment.goals and new_state not in encountered_goals:
			encountered_goals.append(new_state)
			if environment.goals == encountered_goals:
				# reward += 100 
				reward = 20
				reached_allgoals = True
				rchd_order += 1 
				best_path_taken = path_taken
				done = True
			else:
				# something wrong with this code maybe?
				if new_state != environment.goals[len(encountered_goals) - 1]:
					reward = -10
					wrong_order = True
				else:
					reward = 10
					# reward += 50
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
	if running_reward > (log_interval*solved_reward):
		print("########## Solved! ##########")
		torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format(env_name))
		break
		
	# logging
	if i_episode % log_interval == 0:
		avg_length = int(avg_length/log_interval)
		running_reward = int((running_reward/log_interval))		
		print('Episode {} \t Step: {} \t reward: {}'.format(i_episode, step, cumulative_reward))
		print('Number of times agent reached goals in specific order- ', rchd_order)
		print
		running_reward = 0
		avg_length = 0
		
	if highest_reward < cumulative_reward:
		highest_reward = cumulative_reward
		best_path_taken = path_taken

	reward_per_episode.append(cumulative_reward) 
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

print('Number of times agent reached goals in specific order- ', rchd_order)
print('Number of times agent hit an obstacle - ', rchd_obstacles)
print('Min number of steps -', len(best_path_taken))
print('Highest Reward -', highest_reward)

# Printing the time
print('Average time taken to run each trial -', np.mean(time_trials))
print('Max time taken in trial -',np.max(time_trials))

print('Reward at the end of the episode - ',reward_per_episode[len(reward_per_episode) - 1])
print('Average Reward - ', np.mean(reward_per_episode))

end_code = time.time() - starttime_toruncode
print('Total time taken to run the code -', end_code)

plt.plot(reward_per_episode)
plt.xlabel('Episode Number')
plt.ylabel('Rewards')
plt.savefig('single_ag_grid.png')
plt.show()

##########




