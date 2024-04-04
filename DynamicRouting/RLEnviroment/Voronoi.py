import gym
import numpy as np

from DynamicRouting.Core.DynamicRoutingNet import DynamicRoutingNet
from DynamicRouting.RLEnviroment.TaxiPlotter import plot
from DynamicRouting.Utils.RL import RewardAverage
from DynamicRouting.Utils.pathselection import common_path, backup_code
from DynamicSynapse.Adapter.RangeAdapter import RangeAdapter
import matplotlib.pyplot as plt

from DynamicRouting.Utils.ListOperation import onehot
from DynamicSynapse.Utils.loggable import Loggable
import os
import pickle
import InsectGym

np.seterr(invalid='raise')

if __name__ == "__main__":
    goal_provided = True
    if goal_provided:
        environment = 'VoronoiWorldGoal-v1'
        num_goals = 1
    else:
        environment = 'VoronoiWorld-v1'
        num_exits = 2
    task = 'DynamicRouting'
    path_dict = common_path(environment + '_' + task)
    backup_code(source_code_path=os.path.dirname(os.getcwd()), backup_path=path_dict['code_path'])
    Info = 0
    if goal_provided:
        env = gym.make(environment, multi_route_prob=0.1, plot_path=path_dict['plot_path'],
                    task_path=path_dict['task_path'], random_start=False, num_goals=num_goals)
    else:
        env = gym.make(environment, multi_route_prob=0.1, plot_path=path_dict['plot_path'],
                       task_path=path_dict['task_path'], num_exits=1)
    # env = gym.make(environment).env
    env = gym.wrappers.Monitor(env, path_dict['video_path'], video_callable=lambda episode_id: True, force=True)
    env.seed(0)
    env._max_episode_steps = 10000
    max_number_of_episodes = 2000
    highscore = 0
    lastAngle = 0
    WhenRender = 0
    WhenRenderStep = 0.5
    Rendering = 1
    ForceRendering = False
    dt = 0.02
    high_potential_update_period = 1 # was 10000 before20240305  # steps
    reward_adaption = False
    plot_reward = False
    recording = True
    debug = False
    normalise_weight = False
    action_choice = "Probability"  # "Max" #SoftMax "Probability":
    guider_type = 'power'  # 'current' # 'sum'
    exploration = "guided"  # "random" #
    if goal_provided:
        number_of_states = env.observation_space['observation'].n
    else:
        number_of_states = env.observation_space.n
    number_of_action_neurons = env.action_space.n

    step_logger = Loggable()
    step_logger.init_recording(name_list=['i_episode', 'step', 'state', 'reward', 'action', 'cumu_reward'],
                               log_path=path_dict['data_path'], log_name='experiment_step')
    episode_logger = Loggable()
    episode_logger.init_recording(name_list=['i_episode', 'step', 'episode_reward', 'average_episode_reward', 'state',],
                                  log_path=path_dict['data_path'], log_name='experiment_episode')

    reward_averager = RewardAverage(window_size=100)

    DRN = DynamicRoutingNet(number_of_states, dt=dt, default_connection_strength=0,
                            max_connection_strength=1, learning_rate=0.05,                     # learning_rate=0.001,
                            increase_threshold=0, decrease_threshold=0, init_leakage_conductance=0,
                            default_leakage_conductance=0, target_update_rate=0.1)
    DRN.init_second_order_synapse(number_of_action_neurons=number_of_action_neurons, weight=0.01, code_max_only=True,
                                  route_max_only=True, guider_type=guider_type)

    if recording:
        DRN.init_recording(log_path=path_dict['recording_path'], log_name='DRN')

    if reward_adaption:
        reward_adapter = RangeAdapter(targe_max_output=10, targe_min_output=-10, factor=1, bias=0,
                                      update_rate=0.05, t=0, name='reward_adapter')
        if plot_reward:
            reward_adapter.init_recording(log_path=path_dict['data_path'], log_name='reward_adapter')
            reward_figure = None
    cumu_reward = 0

    success_count = 0

    DRN.init_heatmap()

    # DRN.generate_graph()
    # DRN.plot_graph()
    # list(env.decode(209))

    for i_episode in range(max_number_of_episodes):  # run 20 episodes
        observation = env.reset()
        if goal_provided:
            action = DRN.step(onehot(number_of_states, observation['observation']), action_choice=action_choice,
                              debug=debug)[0]
            DRN.reset_target()
            if np.isscalar(observation['desired_goal']):
                DRN.set_target([[observation['desired_goal'], 100]])
            else:
                DRN.set_target([[a_goal,  100] for a_goal in observation['desired_goal']])
        else:
            action = DRN.step(onehot(number_of_states, observation), action_choice=action_choice, debug=debug)[0]
        if exploration == "random":
            action = env.action_space.sample()
        step = 0
        points = 0
        if ForceRendering or Rendering == 1 or i_episode > 1000:
            env.render()
        while True:  # run until episode is done
            step += 1
            observation, reward, done, info = env.step(action)
            last_action = action

            if goal_provided:
                action = DRN.step(onehot(number_of_states, observation['observation']), action_choice=action_choice,
                                  debug=debug, change_high_potential=step % high_potential_update_period == 1)[0]
                DRN.reset_target()
                if np.isscalar(observation['desired_goal']):
                    DRN.set_target([[observation['desired_goal'], 100]])
                else:
                    DRN.set_target([[a_goal, 100] for a_goal in observation['desired_goal']])
                # DRN.set_target([[observation['desired_goal'], 100]])
            else:
                action = DRN.step(onehot(number_of_states, observation), action_choice=action_choice, debug=debug,
                                  change_high_potential=step % high_potential_update_period == 1)[0]
            if exploration == "random":
                action = env.action_space.sample()
            if ForceRendering or Rendering == 1:
                env.render()
            if reward_adaption:
                adapted_reward = reward_adapter.step_dynamics(dt, np.array([reward]))[0]
                if plot_reward:
                    reward_adapter.recording()
                reward_adapter.update()
            else:
                adapted_reward = reward

            if Info:
                print(
                    'episode: %s, step: %s, reward: %.6f, adapted_reward, %.6f, cumu_reward:%.6f, state: %s, action: %s, success_count: %s, targets:'
                    % (i_episode, step, reward, adapted_reward, DRN.cumu_reward, np.argmax(DRN.state_code), action,
                       success_count) + str(
                        DRN.targets))

            cumu_reward += reward
            DRN.record_heatmap()
            DRN.unsupervised_learning(fact_action=last_action, first_step=step == 1, normalise_weight=normalise_weight)
            DRN.reinforcement_learning(reward=adapted_reward, done=done, if_target_state=reward >= 10,
                                       cumu_reward=cumu_reward, first_step=step == 1, synapse_2nd_RL=False)
            # points += reward * np.log(dt + 1)
            step_logger.i_episode = i_episode
            step_logger.cumu_reward = cumu_reward
            step_logger.step = step
            step_logger.state = observation
            step_logger.reward = reward
            step_logger.action = action
            step_logger.recording()

            if done:
                if reward >= 0:
                    success_count += 1
                print(
                    'episode: %s, step: %s, reward: %.6f, adapted_reward, %.6f, cumu_reward:%.6f, state: %s, action: %s, success_count: %s, targets:'
                    % (i_episode, step, reward, adapted_reward, DRN.cumu_reward, np.argmax(DRN.state_code), action,
                       success_count) + str(
                        DRN.targets))
                PointsLast = points
                env.render()
                if Rendering == 1:
                    Rendering = 0
                    WhenRender += WhenRenderStep
                if PointsLast > WhenRender or PointsLast > 30:
                    Rendering = 1
                if recording:
                    DRN.recording()
                    DRN.memory_maintenance(force_save=True)
                step_logger.memory_maintenance(force_save=True)

                average_episode_reward = reward_averager.step(cumu_reward)
                episode_logger.i_episode = i_episode
                episode_logger.episode_reward = cumu_reward
                episode_logger.average_episode_reward = average_episode_reward
                episode_logger.step = step
                episode_logger.state = observation
                episode_logger.recording()
                # env.reset()
                cumu_reward = 0
                ModulatorAmount = 0
                if reward_adaption:
                    if plot_reward:
                        reward_adapter.save_recording(append=True)
                        if reward_figure is None:
                            reward_figure = reward_adapter.plot()['0']
                        else:
                            reward_adapter.plot(figure_handle=reward_figure)
                        reward_adapter.clear_record_cache()
                        plt.grid()
                        plt.show()

                break

            DRN.update()
    episode_logger.save_recording()
    episode_logger.clear_record_cache()
    if goal_provided:
        DRN.generate_graph(number_of_edges=1000)
    else:
        DRN.generate_graph(number_of_edges=25 * 4 * 4 * 5)
    DRN.save_graph(os.path.join(path_dict['model_path'], 'graph.csv'))
    with open(os.path.join(path_dict['model_path'], 'DRN.pkl'), 'wb') as file_DRN:
        pickle.dump(DRN, file_DRN)
    episode_trace = episode_logger.retrieve_record()
    step_trace = step_logger.retrieve_record()
    plot(plot_path=path_dict['plot_path'], episode_trace=episode_trace, DRN=DRN)

'''
    Passenger locations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
    - 4: in taxi
    Destinations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
    Actions:
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: drop off passenger
    Rewards:
    There is a default per-step reward of -1,
    except for delivering the passenger, which is +20,
    or executing "pickup" and "drop-off" actions illegally, which is -10.
    Rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters (R, G, Y and B): locations for passengers and destinations
'''
