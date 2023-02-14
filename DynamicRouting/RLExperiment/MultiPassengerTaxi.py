import gym
import numpy as np

from DynamicRouting.Core.DynamicRoutingNet import DynamicRoutingNet
from DynamicRouting.RLExperiment.TaxiPlotter import plot
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

    environment = 'MultiPassengerTaxi-v1'
    task = 'DynamicRouting'
    path_dict = common_path(environment + '_' + task)
    backup_code(source_code_path=os.path.dirname(os.getcwd()), backup_path=path_dict['code_path'])
    Info = 0
    env = gym.make(environment)
    env = gym.wrappers.Monitor(env, path_dict['video_path'], video_callable=lambda episode_id: True, force=True)
    env.seed(0)
    env._max_episode_steps = 1000
    max_number_of_episodes = 20
    highscore = 0
    lastAngle = 0
    WhenRender = 0
    WhenRenderStep = 0.5
    Rendering = 1
    ForceRendering = True
    dt = 0.02
    reward_adaption = False
    plot_reward = False
    map_only = False
    recording = True
    debug = False
    action_choice = "Probability"  # "Max" #SoftMax "Probability":
    guider_type = 'power'  # 'current' # 'sum'
    exploration = "guided"  # "random" #
    number_of_states = 10
    if map_only:
        number_of_states = 25
    else:
        number_of_states = env.observation_space.n
    number_of_action_neurons = env.action_space.n

    step_logger = Loggable()
    step_logger.init_recording(name_list=['i_episode', 'step', 'state', 'reward', 'action', 'cumu_reward',
                                     'taxi_row', 'taxi_col', 'pass_on'],
                               log_path=path_dict['data_path'], log_name='experiment_step')
    episode_logger = Loggable()
    episode_logger.init_recording(name_list=['i_episode', 'step', 'episode_reward', 'average_episode_reward'],
                               log_path=path_dict['data_path'], log_name='experiment_episode')

    reward_averager = RewardAverage(window_size=100)

    dest = np.array([0, 6])
    dest_state = dest[0] * 7 + dest[1]

    DRN = DynamicRoutingNet(number_of_states, dt=dt, default_connection_strength=0,
                            max_connection_strength=1, learning_rate=0.001,
                            increase_threshold=0, decrease_threshold=0, init_leakage_conductance=0,
                            default_leakage_conductance=0, target_update_rate=0.1)
    DRN.init_second_order_synapse(number_of_action_neurons=number_of_action_neurons, weight=0.01, guider_type=guider_type)

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

    for i_episode in range(max_number_of_episodes):  # run 20 episodes
        observation = env.reset()
        if map_only:
            taxi_row, taxi_col, pass_on = env.decode(observation)
            location_state = taxi_row * 7 + taxi_col
        if map_only:
            action = DRN.step(onehot(number_of_states, location_state), action_choice=action_choice, debug=debug)[0]
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
            taxi_row, taxi_col, pass_on = env.decode(observation)
            if map_only:
                action = DRN.step(onehot(number_of_states, location_state), action_choice=action_choice, debug=debug)[0]
            else:
                action = DRN.step(onehot(number_of_states, observation), action_choice=action_choice, debug=debug)[0]
            if exploration == "random":
                action = env.action_space.sample()
            location_state = taxi_row * 7 + taxi_col
            if map_only:
                target_state = dest_state
                DRN.reset_target()
                DRN.set_target([[target_state, 100]])
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
                print('new observation: %s, taxi_row: %s, taxi_col: %s, pass_on: %s' % (
                    observation, taxi_row, taxi_col, pass_on))

            cumu_reward += reward
            DRN.record_heatmap()
            DRN.unsupervised_learning(fact_action=last_action, first_step=step==1)
            DRN.reinfrocement_learning(reward=adapted_reward, done=done, if_target_state=reward >= 15,
                                       cumu_reward=cumu_reward, first_step=step==1)

            step_logger.i_episode = i_episode
            step_logger.cumu_reward = cumu_reward
            step_logger.step = step
            step_logger.state = observation
            step_logger.reward = reward
            step_logger.action = action
            step_logger.taxi_row = taxi_row
            step_logger.taxi_col = taxi_col
            step_logger.pass_on = pass_on
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
                if PointsLast > WhenRender or PointsLast >= 3:
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
    if map_only:
        DRN.generate_graph(number_of_edges=1000)
    else:
        DRN.generate_graph(number_of_edges=25*4*4*5)
    DRN.save_graph(os.path.join(path_dict['model_path'], 'graph.csv'))
    with open(os.path.join(path_dict['model_path'], 'DRN.pkl'), 'wb') as file_DRN:
        pickle.dump(DRN, file_DRN)
    episode_trace = episode_logger.retrieve_record()
    step_trace = step_logger.retrieve_record()
    plot(plot_path=path_dict['plot_path'], episode_trace=episode_trace, DRN=DRN)
