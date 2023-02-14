import os

import numpy as np
import matplotlib.pyplot as plt
import gym
import random
from IPython.display import clear_output
from time import sleep
from DynamicRouting.Utils.RL import RewardAverage
from DynamicRouting.RLExperiment.TaxiPlotter import plot
from DynamicSynapse.Utils.loggable import Loggable
from DynamicRouting.Utils.pathselection import common_path, backup_code
import InsectGym

def state_on_goals(num_obs, obs, goals ):
    return num_obs * goals + obs

Test = False
# environment = 'Taxi-v3'
# environment = 'VoronoiWorld-v1'
environment = 'VoronoiWorldGoal-v1'
task = 'QLearning'
path_dict = common_path(environment + '_' + task)
backup_code(source_code_path=os.path.dirname(os.getcwd()), backup_path=path_dict['code_path'])
# CREATE THE ENVIRONMENT
if environment == 'VoronoiWorldGoal-v1' or environment == 'VoronoiWorld-v1':
    env = gym.make(environment, multi_route_prob=0.1, plot_path=path_dict['plot_path'],
                   task_path=path_dict['task_path'], random_start=False).env
else:
    env = gym.make(environment).env
action_size = env.action_space.n
if environment == 'VoronoiWorldGoal-v1':
    obs_size = env.observation_space['observation'].n
    num_goals = env.observation_space['desired_goal'].n
    state_size = obs_size * num_goals
else:
    state_size = env.observation_space.n
print("Action space size: ", action_size)
print("State space size: ", state_size)

# INITIALISE Q TABLE TO ZERO
Q = np.zeros((state_size, action_size))

# HYPERPARAMETERS
train_episodes = 2000  # Total train episodes
test_episodes = 100  # Total test episodes
max_steps = 10000  # Max steps per episode
alpha = 0.7  # Learning rate
gamma = 0.618  # Discounting rate

# EXPLORATION / EXPLOITATION PARAMETERS
epsilon = 1  # Exploration rate
max_epsilon = 1  # Exploration probability at start
min_epsilon = 0.01  # Minimum exploration probability
decay_rate = 0.01  # Exponential decay rate for exploration prob

# TRAINING PHASE
episode_trace = {'episode_reward': [],
                 'average_episode_reward': [],
                 'step': [],
                 }

episode_logger = Loggable()
episode_logger.init_recording(name_list=['i_episode', 'step', 'episode_reward', 'average_episode_reward'],
                              log_path=path_dict['data_path'], log_name='experiment_episode')

# training_rewards = []   # list of rewards
# average_episode_rewards = []
# number_of_steps = []
reward_averager = RewardAverage(window_size=100)

for episode in range(train_episodes):
    if environment == 'VoronoiWorldGoal-v1':
        obs = env.reset()
        state = state_on_goals(obs_size, obs['observation'], obs['desired_goal'])
    else:
        state = env.reset()  # Reset the environment
    cumulative_training_rewards = 0

    for step in range(max_steps):
        # Choose an action (a) among the possible states (s)
        exp_exp_tradeoff = random.uniform(0, 1)  # choose a random number

        # If this number > epsilon, select the action corresponding to the biggest Q value for this state (Exploitation)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(Q[state, :])
        # Else choose a random action (Exploration)
        else:
            action = env.action_space.sample()


        if environment == 'VoronoiWorldGoal-v1':
            obs, reward, done, info = env.step(action)
            new_state = state_on_goals(obs_size, obs['observation'], obs['desired_goal'])
        else:
            # Perform the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, done, info = env.step(action)
        if done:
            step_reward = reward
        else:
            step_reward = 0

            # Update the Q table using the Bellman equation: Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        Q[state, action] = Q[state, action] + alpha * (step_reward + gamma * np.max(Q[new_state, :]) - Q[state, action])
        cumulative_training_rewards += reward  # increment the cumulative reward
        state = new_state  # Update the state

        # If we reach the end of the episode
        if done == True:
            print("Cumulative reward for episode {}: {}. Total steps = {}".format(episode, cumulative_training_rewards, step))
            episode_trace['step'].append(step)
            break

    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)


    # append the episode cumulative reward to the list
    # episode_trace['episode_reward'].append(cumulative_training_rewards)
    average_episode_reward = reward_averager.step(cumulative_training_rewards)
    # episode_trace['average_episode_reward'].append(average_episode_reward)

    episode_logger.i_episode = episode
    episode_logger.episode_reward = cumulative_training_rewards
    episode_logger.average_episode_reward = average_episode_reward
    episode_logger.step = step
    episode_logger.recording()

    episode_logger.save_recording()
    episode_logger.clear_record_cache()

print("Training score over time: " + str(sum(episode_trace['episode_reward']) / train_episodes))

episode_trace = episode_logger.retrieve_record()
plot(plot_path=path_dict['plot_path'],   episode_trace=episode_trace, DRN=None)


# TEST PHASE

if Test:
    test_rewards = []
    frames = []  # for animation

    for episode in range(test_episodes):
        state = env.reset()
        cumulative_test_rewards = 0
        print("****************************************************")
        print("EPISODE ", episode)

        for step in range(max_steps):
            env.render()  # UNCOMMENT IT IF YOU WANT TO SEE THE AGENT PLAYING
            # Take the action (index) that have the maximum expected future reward given that state
            action = np.argmax(Q[state, :])
            new_state, reward, done, info = env.step(action)
            cumulative_test_rewards += reward
            state = new_state

            # Put each rendered frame into dict for animation
            frames.append({
                'frame': env.render(mode='ansi'),
                'state': state,
                'action': action,
                'reward': reward
            }
            )

            if done:
                print("Cumulative reward for episode {}: {}".format(episode, cumulative_test_rewards))
                break
        test_rewards.append(cumulative_test_rewards)

    env.close()
    print("Test score over time: " + str(sum(test_rewards) / test_episodes))


def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'].getvalue())
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)


print_frames(frames)
