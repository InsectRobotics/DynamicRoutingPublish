import copy
import os
import pickle

import gym
import numpy as np
from DynamicSynapse.Utils.loggable import Loggable
from DynamicSynapse.Utils.tracereader import TraceReader
from matplotlib.backends.backend_pdf import PdfPages

from DynamicRouting.Core.DynamicRoutingNet import DynamicRoutingNet
from DynamicRouting.Utils.RL import RewardAverage
from DynamicRouting.Utils.pathselection import common_path, backup_code
from DynamicSynapse.Adapter.RangeAdapter import RangeAdapter
import matplotlib.pyplot as plt
from DynamicRouting.Utils.ListOperation import onehot
import InsectGym

'''
    training regimes:   amylacetate (AM),   1-octanol (OCT)
    odour states:        none,   amylacetate (AM),   1-octanol (OCT)
    reinforcer states:  none,   fructose,   quinine hemisulphate


    Assume there is a linear petri dish with 5 different locations. 
    If AM exists, it always on the left two locations on the petri dish, and there is a gradient from left to right 
    for the maggot chemotaxis. 
    Similarly, if OCT exists, it always on the right two locations of the petri dish, and there is a gradient from 
    right to left for the maggot chemotaxis

    A maggot can choose from doing nothing, appetizing to or escaping from a specific odour.

    As we think there are MBONs in charge of appetizing or escaping behaviour but not for specific odours, there 
    should be other circuits control which odour the actions are performed on. Here for simplification, we 
    treat different combinations of odour and behaviours as individual actions. 

    Hence, there are 1+2X2=5 actions.

    The environment returns the odour and the reinforcer of the location the maggot is on. However, only the odour can
    be perceived given the location. The reinforcer is contained in agarose, hence the maggot can perceive it only 
    immediately after an appetizing behaviour, which consists of a chemotaxis behaviour and a forage behaviour.
'''


def reward_to_reinforcer(reward):
    if reward == 1:
        reinforcer = 'fructose'
    elif reward == -1:
        reinforcer = 'quinine'
    else:
        reinforcer = None
    return reinforcer


def maggot_state_encoder(perceived_odour, reward):
    reinforcer = reward_to_reinforcer(reward)

    if reinforcer == 'quinine':
        state_code = 5
    elif reinforcer == 'fructose':
        state_code = 4
    elif perceived_odour == 'MIX':
        state_code = 3
    elif perceived_odour == 'OCT':
        state_code = 2
    elif perceived_odour == 'AM':
        state_code = 1
    else:
        state_code = 0
    return state_code


class Maggot(DynamicRoutingNet):
    def __init__(self, *args, **kwargs):
        DynamicRoutingNet.__init__(self, *args, **kwargs)
        self.name = 'Maggot'
        self.state_code_index = 0

    def state_encoder(self, obs):
        perceived_odour, reward = obs
        self.state_code_index = maggot_state_encoder(perceived_odour, reward)

    def step(self, *args, **kwargs):
        action = DynamicRoutingNet.step(self, onehot(self.number_of_neurons, self.state_code_index), *args, **kwargs)
        return action

def episode_experiment(env, maggot, i_episode=0, maxstep=20, step_logger=None, episode_logger=None, i_maggot=0,
                       i_protocol=0, normalise_weight=False, route_exploration=True,learning_rate= 0., if_unexpected_learning=False):
    observation = env.reset()
    odour, reinforcer, location = env.decode(observation)
    reward = 0
    done = False
    action = None
    step = 0
    for i in range(maxstep):  # run until episode is done
        step += 1
        env.render()
        maggot.update()
        maggot.state_encoder((odour, reward))
        if reward:
            print('reward: %.6f, new_state: %d' %(reward, maggot.state_code_index))

        last_action = action
        action = maggot.step(route_exploration=route_exploration)[0]
        if action == 1:
            print("step: "+str(step))
        maggot.record_heatmap()
        # maggot.synapse.check_decrease(mode='New')
        maggot.unsupervised_learning(fact_action=last_action, first_step=step == 1,  normalise_weight=normalise_weight,
                                     if_unexpected_learning=if_unexpected_learning, learning_rate=learning_rate)
        # maggot.synapse.check_decrease(mode='')
        maggot.reinforcement_learning(fact_action=last_action, reward=reward, done=done, if_target_state=reward >= 1,
                                      first_step=step==1, synapse_2nd_RL=False)
                                      # if_danger_state=reward <= -1, first_step=step == 1)
        # maggot.synapse.check_decrease(mode='')
        print(
            'i_protocol:%d, i_maggot:%d, episode: %s, step: %s, reward: %.6f, cumu_reward:%.6f, state: %s, action: %s, sub_target: %d, targets:'
            % (i_protocol, i_maggot, i_episode, step, reward, maggot.cumu_reward, maggot.state_code_index,
               action, maggot.synapse.max_route_index) + str(
                maggot.targets))

        if Info:
            print('info', info)
            print('Observation \n', observation)
            print('maggot.potential_vector\n', maggot.potential_vector)
            print('maggot.inject_currents\n', maggot.conservation_vector[:maggot.number_of_neurons])
        # maggot.generate_graph()
        # maggot.plot_graph()
        pass


        if done:
            # last_action = action
            # action = maggot.step()[0]
            # maggot.record_heatmap()
            # maggot.unsupervised_learning(fact_action=last_action, first_step=step == 1,
            #                              normalise_weight=normalise_weight)
            # maggot.reinforcement_learning(fact_action=last_action, reward=reward, done=done,
            #                               if_target_state=reward >= 1,
            #                               first_step=step == 1)
            # print(
            #     'i_protocol:%d, i_maggot:%d, episode: %s, step: %s, reward: %.6f, cumu_reward:%.6f, state: %s, action: %s, sub_target: %d, targets:'
            #     % (i_protocol, i_maggot, i_episode, step, reward, maggot.cumu_reward, maggot.state_code_index,
            #        action, maggot.synapse.max_route_index) + str(
            #         maggot.targets))
            break
        else:
            observation, reward, done, info = env.step(action)
            odour, reinforcer, location = env.decode(observation)


        if recording:
            maggot.recording()
            maggot.memory_maintenance(force_save=False)
            maggot.synapse.recording()
            maggot.synapse.memory_maintenance(force_save=False)

        if step_logger is not None:
            step_logger.i_episode = i_episode
            step_logger.step = step
            step_logger.observation = observation
            step_logger.reward = reward
            step_logger.action = action
            step_logger.odour = odour
            step_logger.reinforcer = reinforcer
            step_logger.sub_target = maggot.synapse.max_route_index
            step_logger.reward = reward
            step_logger.recording()
            step_logger.memory_maintenance(force_save=False)
    else:
        env.stats_recorder.save_complete()
        env.stats_recorder.done = True

    if episode_logger is not None:
        episode_logger.i_episode = i_episode
        episode_logger.maggot = maggot
        episode_logger.num_steps = step  # TODO check if the step is the final number of steps
        episode_logger.final_location = location
        episode_logger.recording()
    # maggot.generate_graph()
    # maggot.plot_graph()
    pass


def plot(plot_path='', episode_trace=None, DRN=None):
    figure_dict = dict()
    # figure_dict['episode_reward'] = plt.figure()
    # figure12lines1, = plt.plot(episode_trace['episode_reward'])
    # plt.legend([figure12lines1], ['Episode Reward'], loc=4)
    # plt.xlabel('Episode')
    # plt.title('Episode Reward')
    # plt.grid()

    if DRN is not None:
        figure_dict['DRN_graph'] = DRN.plot_graph()
        figure_dict['DRN_heatmap'] = DRN.plot_heatmap()

    if plot_path:
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        pp = PdfPages(os.path.join(plot_path, "Emviroment.pdf"))
        for key in figure_dict:
            figure_dict[key].savefig(pp, format='pdf')
        pp.close()

    plt.show()
    return figure_dict


def analyse_plot(trace_file_folder):
    global i_protocol, i_maggot
    # trace_file_folder = "F:\\recording\\MaggotInPetriDish-v1_DynamicRouting\\2023-09-15_23-14-25\\trace\\"
    step_traces = [[None for _ in range(number_of_maggots)] for _ in range(len(protocol))]
    step_locations = [[None for _ in range(number_of_maggots)] for _ in range(len(protocol))]
    location_counts_per_maggot = [[None for _ in range(number_of_maggots)] for _ in
                                  range(len(protocol))]  # numbers of maggots on AM side or on OCT side
    location_counts_per_step = [[None for _ in range(maxstep)] for _ in range(len(protocol))]
    learning_pref_per_maggot = np.zeros((len(protocol), number_of_maggots))
    learning_pref_per_step = np.zeros((len(protocol), maxstep))
    learning_index = {}
    for i_protocol in range(len(protocol)):
        for i_maggot in range(number_of_maggots):
            log_file_path = os.path.join(trace_file_folder,
                                         "protocol%d_maggot%d_experiment_step.pkl" % (i_protocol, i_maggot))
            aTR = TraceReader(log_file_path=log_file_path)
            step_trace = aTR.get_trace()

            step_traces[i_protocol][i_maggot] = step_trace  # = step_loggers[i_protocol][i_maggot].retrieve_record()
            # for i_episode in range(max(step_trace['i_episode'])):
            #     i_episode_step_index = step_trace['i_episode'] == i_episode
            #     episode_step_trace = step_trace['observation'][i_episode_step_index]
            #     pref = (episode_step_trace.count['0'] + episode_step_trace.count['1']
            #             - episode_step_trace.count['3'] - episode_step_trace.count['4'])/len(episode_step_trace)
            # final_locations[i_protocol][i_maggot] = step_trace['observation'][-1]
            step_locations[i_protocol][i_maggot] = step_trace['observation']

            locations, counts = np.unique(step_trace['observation'], return_counts=True)
            print('locations')
            print(locations)

            location_counts_per_maggot[i_protocol][i_maggot] = counts
            print('location_counts_per_maggot[%d][%d]' % (i_protocol, i_maggot))
            print(location_counts_per_maggot[i_protocol][i_maggot])
            # # AM side  counts[0:2] # # OCT side counts[3:5]
            learning_pref_per_maggot[i_protocol][i_maggot] = (sum(counts[0:2]) - sum(counts[3:5])) / maxstep
            print('learning_pref_per_maggot[%d][%d]' % (i_protocol, i_maggot))
            print(learning_pref_per_maggot[i_protocol][i_maggot])
        # # AM side
        # maggot_count[i_protocol, 0] = final_locations[i_protocol].count(0) + final_locations[i_protocol].count(1)
        # # OCT side
        # maggot_count[i_protocol, 1] = final_locations[i_protocol].count(3) + final_locations[i_protocol].count(4)
    # print('maggot_count')
    # print(maggot_count)
    # learning_pref = (maggot_count[:, 0] - maggot_count[:, 1]) / number_of_maggots
    step_locations = np.array(step_locations)
    for i_protocol in range(len(protocol)):
        for i_step in range(maxstep):
            locations, counts = np.unique(step_locations[i_protocol, :, i_step])
            location_counts_per_step[i_protocol][i_step] = counts
            learning_pref_per_step[i_protocol][i_step] = counts
            print('learning_pref_per_step[%d][%d]' % (i_protocol, i_step))
            print(location_counts_per_maggot[i_protocol][i_step])
            # # AM side  counts[0:2] # # OCT side counts[3:5]
            learning_pref_per_step[i_protocol][i_step] = (counts[0:2] - counts[3:5]) / number_of_maggots
            print('learning_pref_per_step[%d][%d]' % (i_protocol, i_step))
            print(learning_pref_per_step[i_protocol][i_step])
    learning_pref_per_step = np.array(learning_pref_per_step)
    print('learning_pref_per_step')
    print(learning_pref_per_step)
    learning_index['FN'] = (learning_pref_per_step[0] - learning_pref_per_step[1]) / 2
    learning_index['FF'] = (learning_pref_per_step[2] - learning_pref_per_step[3]) / 2
    learning_index['QN'] = (learning_pref_per_step[4] - learning_pref_per_step[5]) / 2
    learning_index['QQ'] = (learning_pref_per_step[6] - learning_pref_per_step[7]) / 2
    print(learning_index)
    labels = ['FN', 'FF', 'QN', 'QQ']
    learning_index_array = np.hstack(learning_index['FN'],
                                     learning_index['FF'],
                                     learning_index['QN'],
                                     learning_index['QQ'])
    fig, ax = plt.subplots(nrows=1, ncols=1)
    bplot = ax.boxplot(learning_index,
                       notch=False,  # notch shape
                       vert=True,  # vertical box alignment
                       patch_artist=True,  # fill with color
                       labels=labels)  # will be used to label x-ticks
    ax.set_title('learning_pref_per_step')
    ax.yaxis.grid(True)
    ax.set_xlabel('Three separate samples')
    ax.set_ylabel('learning pref per step')


if __name__ == "__main__":

    environment = 'MaggotInPetriDish-v1'
    task = 'DynamicRouting'
    path_dict = common_path(environment + '_' + task)
    backup_code(source_code_path=os.path.dirname(os.getcwd()), backup_path=path_dict['code_path'])
    maxstep = 40
    Info = 0
    never_done = True
    # protocol = [{'odor': ['AM', 'AM'], 'reinforcer': ['fructose']},
    #             {'odor': ['OCT', 'OCT'], 'reinforcer': [None]},
    #             {'odor': ['AM', 'AM'], 'reinforcer': ['fructose']},
    #             {'odor': ['OCT', 'OCT'], 'reinforcer': [None]},
    #             {'odor': ['AM', 'AM'], 'reinforcer': ['fructose']},
    #             {'odor': ['OCT', 'OCT'], 'reinforcer': [None]},
    #             {'odor': ['AM', 'OCT'], 'reinforcer': ['fructose']}]

    pretri_dishes = {
        'AAN': {'odor': ['AM', 'AM'], 'reinforcer': [None]},
        'AAF': {'odor': ['AM', 'AM'], 'reinforcer': ['fructose']},
        'AAQ': {'odor': ['AM', 'AM'], 'reinforcer': ['quinine']},
        'OON': {'odor': ['OCT', 'OCT'], 'reinforcer': [None]},
        'OOF': {'odor': ['OCT', 'OCT'], 'reinforcer': ['fructose']},
        'OOQ': {'odor': ['OCT', 'OCT'], 'reinforcer': ['quinine']},
        'AON': {'odor': ['AM', 'OCT'], 'reinforcer': [None]},
        'AOF': {'odor': ['AM', 'OCT'], 'reinforcer': ['fructose']},
        'AOQ': {'odor': ['AM', 'OCT'], 'reinforcer': ['quinine']},
    }
    training = {
        "AM+/OCT": ['AAF', 'OON', 'AAF', 'OON', 'AAF', 'OON'],
        "AM/OCT+": ['OOF', 'AAN', 'OOF', 'AAN', 'OOF', 'AAN'],
        "AM-/OCT": ['AAQ', 'OON', 'AAQ', 'OON', 'AAQ', 'OON'],
        "AM/OCT-": ['OOQ', 'AAN', 'OOQ', 'AAN', 'OOQ', 'AAN'],
    }

    testing = ['AON', 'AOF', 'AOQ']

    protocol = [
        {'training': training['AM+/OCT'], 'testing': 'AON'},
        {'training': training['AM/OCT+'], 'testing': 'AON'},
        {'training': training['AM+/OCT'], 'testing': 'AOF'},
        {'training': training['AM/OCT+'], 'testing': 'AOF'},
        {'training': training['AM-/OCT'], 'testing': 'AON'},
        {'training': training['AM/OCT-'], 'testing': 'AON'},
        {'training': training['AM-/OCT'], 'testing': 'AOQ'},
        {'training': training['AM/OCT-'], 'testing': 'AOQ'},
    ]

    number_of_maggots = 30

    envs = {}
    for a_key, an_item in pretri_dishes.items():
        env = gym.make(environment, odor=an_item['odor'], reinforcer=an_item['reinforcer'], num_locations=5,
                       never_done=never_done)
        env = gym.wrappers.Monitor(env, os.path.join(path_dict['video_path'], a_key),
                                   video_callable=lambda episode_id: True, force=True)
        envs[a_key] = env
    dt = 0.02
    plot_reward = False
    recording = True
    debug = False
    action_choice = "Probability"  # "Max" #SoftMax "Probability":
    guider_type = 'current' # 'power'  # 'sum'
    exploration = "guided"  # "random" #
    number_of_states = 6
    number_of_action_neurons = 5
    normalise_weight = False # False
    route_exploration = False # True
    # learning_rate = 0.05
    learning_rate = 1e-3
    if_unexpected_learning = True

    maggots = [[None for _ in range(number_of_maggots)] for _ in range(len(protocol))]
    step_loggers = [[None for _ in range(number_of_maggots)] for _ in range(len(protocol))]
    episode_loggers = [[None for _ in range(number_of_maggots)] for _ in range(len(protocol))]
    for i_protocol in range(len(protocol)):
        number_of_episode = len(protocol[i_protocol]['training']) + 1
        for i_maggot in range(number_of_maggots):
            # maggot = Maggot(number_of_states, dt=dt, default_connection_strength=1e-10, learning_rate=0.05,
            #                 increase_threshold=0, decrease_threshold=0, init_leakage_conductance=1e-5,
            #                 default_leakage_conductance=1e-10, target_update_rate=0.1)
            # maggot = Maggot(number_of_states, dt=dt, default_connection_strength=0, learning_rate=0.01,
            #                 increase_threshold=0, decrease_threshold=0, init_leakage_conductance=0,
            #                 default_leakage_conductance=0, target_update_rate=0.1)
            # maggot = Maggot(number_of_states, dt=dt, default_connection_strength=0, learning_rate=0.05, #0.05,
            #                 increase_threshold=0, decrease_threshold=0, init_leakage_conductance=1e-10,
            #                 default_leakage_conductance=1e-10, target_update_rate=0.1)

            maggot = Maggot(number_of_states, dt=dt, default_connection_strength=0, learning_rate=0.1, #0.05,
                            increase_threshold=0, decrease_threshold=0, init_leakage_conductance=0,
                            default_leakage_conductance=0, target_update_rate=0.1)
            second_order_synapse_weights = np.zeros((number_of_states, number_of_states, number_of_action_neurons))
            # for i in range(number_of_states):
            #     for j in range(number_of_states):
            #         for k in range(number_of_action_neurons):
            #             second_order_synapse_weights[i, j, k] = 0.1*i+0.01*j+0.001*k
            maggot.init_second_order_synapse(number_of_action_neurons=number_of_action_neurons, weight=1e-10)
            maggot.synapse.init_recording(log_path=path_dict['data_path'],
                                  log_name='2ndOrderDynapseProtocol%d_maggot%d' % (i_protocol, i_maggot))
            maggot.init_heatmap()
            maggot.init_recording(log_path=path_dict['data_path'],
                                  log_name='protocol%d_maggot%d' % (i_protocol, i_maggot))
            maggots[i_protocol][i_maggot] = maggot

            step_logger = Loggable()
            step_logger.init_recording(name_list=['i_episode', 'step', 'observation', 'reward', 'sub_target',
                                                  'action', 'odour',
                                                  'reinforcer', 'reward'],
                                       log_path=path_dict['data_path'],
                                       log_name='protocol%d_maggot%d_experiment_step' % (i_protocol, i_maggot))
            step_loggers[i_protocol][i_maggot] = step_logger

            episode_logger = Loggable()
            episode_logger.init_recording(name_list=['i_episode', 'maggot', 'num_steps', 'final_location'],
                                          log_path=path_dict['data_path'], log_name='experiment_episode')
            episode_loggers[i_protocol][i_maggot] = episode_loggers

            for i_episode in range(number_of_episode):
                if i_episode < len(protocol[i_protocol]['training']):
                    episode_experiment(envs[protocol[i_protocol]['training'][i_episode]], maggot, i_episode=i_episode,
                                       maxstep=maxstep, step_logger=step_logger,
                                       episode_logger=episode_logger, i_maggot=i_maggot, i_protocol=i_protocol,
                                       normalise_weight=normalise_weight, route_exploration=route_exploration,
                                       learning_rate= learning_rate, if_unexpected_learning=if_unexpected_learning)
                else:
                    episode_experiment(envs[protocol[i_protocol]['testing']], maggot, i_episode=i_episode,
                                       maxstep=maxstep, step_logger=step_logger,
                                       episode_logger=episode_logger, i_maggot=i_maggot, i_protocol=i_protocol,
                                       normalise_weight=normalise_weight, route_exploration=route_exploration,
                                       learning_rate= learning_rate, if_unexpected_learning=if_unexpected_learning)
            maggot.save_recording()
            maggot.synapse.save_recording()
            maggot.clear_record_cache()
            maggot.synapse.clear_record_cache()
            maggot.generate_graph()
            maggot.save_graph(
                os.path.join(path_dict['model_path'], 'protocol%d_maggot%d_graph' % (i_protocol, i_maggot)))
            # maggot.plot_graph()
            # maggot.plot_heatmap()
            step_logger.save_recording()
            step_logger.clear_record_cache()
            episode_logger.save_recording()
            episode_logger.clear_record_cache()

    with open(os.path.join(path_dict['model_path'], 'maggots.pkl'), 'wb') as file_DRN:
        pickle.dump(maggots, file_DRN)

    # episode_trace = episode_logger.retrieve_record()
    # step_trace = step_logger.retrieve_record()

    # analyse_plot()

    print(path_dict['plot_path'])
    pass