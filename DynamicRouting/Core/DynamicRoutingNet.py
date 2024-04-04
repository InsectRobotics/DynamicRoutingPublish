import copy
import os
import warnings

import numpy as np
from DynamicSynapse.Utils.MatToVideo2 import matrix2video2

from DynamicRouting.StateCoder.DirectGraphBuilder import LearningDirectedNeuronGraph
from DynamicRouting.Core.AsymmetricalNodalCircuitExtraGround import \
    AsymmetricalNodalCircuitExtraGround as AsymmetricalNodalCircuit
# from DynamicRouting.AsymmetricalNodalCircuitExtraGroundAndSource import \
#     AsymmetricalNodalCircuitExtraGroundAndSource as AsymmetricalNodalCircuit

from DynamicRouting.Core.SecondOrderSynapseSoftmax import SecondOrderSynapsSoftmax as SecondOrderSynapse
from DynamicRouting.StateCoder.FullRandomMapping import FullRandomMapping
from DynamicRouting.Utils.MathUtil import softmax, positive_variable_update, relu, sigmoid
from DynamicRouting.Utils.ListOperation import onehot
from DynamicSynapse.Utils.plotscripts import heatmap2d
import matplotlib.pyplot as plt
from DynamicSynapse.Utils.loggable import Loggable


def auto_layout(number_of_plots):
    sqrt = int(np.ceil(np.sqrt(number_of_plots)))
    for i in reversed(range(sqrt)):
        if number_of_plots % (i + 1) == 0:
            number_of_row = i + 1
            break
    if number_of_row == 1:
        number_of_row = sqrt
    number_of_column = int(np.ceil(number_of_plots / number_of_row))
    return number_of_row, number_of_column

class DynamicRoutingNet(LearningDirectedNeuronGraph, AsymmetricalNodalCircuit, Loggable):
    def __init__(self, number_of_neurons, default_connection_strength=1e-10, max_connection_strength=10,
                 learning_rate=1e-5, dt=0.01,
                 increase_threshold=0, decrease_threshold=0, init_leakage_conductance=1e-5,
                 default_leakage_conductance=1e-10,
                 target_update_rate=0.1, target_decay_rate=0.1, if_single_action=True, code_max_only=True,
                 one_target=False, max_ground_conductance=1):
        LearningDirectedNeuronGraph.__init__(self, number_of_neurons, default_connection_strength,
                                             max_connection_strength, learning_rate,
                                             dt, increase_threshold, decrease_threshold, code_max_only=code_max_only)
        AsymmetricalNodalCircuit.__init__(self, number_of_neurons, default_connection_strength,
                                          init_leakage_conductance)
        # AsymmetricalNodalCircuit puts init_leakage_conductance to self.default_leakage_conductance
        self.learning_rate = learning_rate
        self.default_leakage_conductance = default_leakage_conductance
        self.target_update_rate = target_update_rate
        self.target_decay_rate = target_decay_rate
        self.state_code = np.zeros(self.number_of_neurons)
        self.state_code_last = np.zeros(self.number_of_neurons)
        self.cumu_reward = 0.0
        self.last_cumu_reward = copy.deepcopy(self.cumu_reward)
        self.name_list_log = ['connection_matrix', 'leakage_conductance', 'targets', 'potential_conservations',
                              'conservation_vector', 'state_code']
        self.update_action_flag = True
        self.if_single_action = if_single_action
        self.one_target = one_target
        self.max_ground_conductance = max_ground_conductance

    def init_full_random_mapping(self, number_of_inputs, threshold=5, activity_check_start_time=20,
                                 code_max_only=True):
        self.state_coder_random = FullRandomMapping(number_of_inputs, self.number_of_neurons, dt=self.dt, threshold=threshold,
                                                    activity_check_start_time=activity_check_start_time,
                                                    code_max_only=code_max_only)

    def init_second_order_synapse(self, number_of_action_neurons, weight=None, preference='', code_max_only=True,
                                  route_max_only=True, guider_type='power'):
        self.synapse = SecondOrderSynapse(number_of_state_neurons=self.number_of_neurons,
                                          number_of_action_neurons=number_of_action_neurons,
                                          weight=weight, preference=preference, code_max_only=code_max_only,
                                          route_max_only=route_max_only, if_single_action=self.if_single_action,
                                          guider_type=guider_type)

    def step(self, obs, dt=None, code_max_only=True, potential=True, action_choice="SoftMax", debug=False, top_n=5,
             action_noise_variance=0, route_exploration=False, change_high_potential=True):
        if dt is None:
            dt = self.dt
        self.obs = obs
        if hasattr(self, 'state_coder_random'):
            self.state_code = self.state_coder_random.step(obs, dt, update_mapping=False)
        else:
            self.state_code = obs
        # self.update_inject_currents_by_vector((self.state_code-self.state_code_last)*self.default_connection_strength)
        if change_high_potential:
            if code_max_only:
                activating_state = np.argmax(self.state_code)
                if potential:
                    self.reset_potential_conservations()
                    self.add_potential_conservations([[activating_state, -1, 1]])
                else:
                    inject_currents = np.zeros(self.number_of_neurons)
                    if self.state_code[activating_state] <= 0:
                        warnings.warn("No positive activating state")
                    inject_currents[activating_state] = self.state_code[activating_state] * self.connection_matrix.mean() * \
                                                        self.number_of_neurons
                    self.set_inject_currents_by_vector(inject_currents)
            else:
                raise NotImplementedError
        for _ in range(2):
            self.update_potential_diff()
            self.update_nodal_matrix()
            self.solve()
            self.update_current()
        self.action_neuron_current = self.synapse.step(self.potential_vector, self.current_matrix, code_max_only=True,
                                                       state_code=self.state_code, route_exploration=route_exploration,
                                                       guider_noise_variance=action_noise_variance)
        self.action_neuron_potential = relu(1 * self.action_neuron_current)
        # self.action = np.random.choice(self.synapse.number_of_action_neurons, 1,
        #                                p=softmax(self.action_neuron_potential))
        if code_max_only:
            # I can't remember why I set the condition about last indexes.
            '''# if self.synapse.actual_state_index_present != self.synapse.actual_state_index_last or \
            #     self.synapse.max_route_index_last != self.synapse.max_route_index_last:
            #     self.update_action_flag = True
            #     if self.if_single_action:
            #         self.action = self.single_action_choose(action_choice)
            #     else:
            #         self.action = self.vector_action_noise(variance=action_noise_variance)
            # else:
            #     self.update_action_flag = False
            '''
            if self.if_single_action:
                self.action = self.single_action_choose(action_choice)
            else:
                self.action = self.vector_action_noise(variance=action_noise_variance)
            self.update_action_flag = True
        else:
            raise NotImplementedError
        if debug:
            print("self.action_neuron_potential before postprocessing")
            print(self.action_neuron_potential)

        return self.action


    def vector_action_noise(self, variance=0):
        self.action = self.action_neuron_potential \
                    + sigmoid(np.random.normal(np.zeros_like(self.action_neuron_potential), variance))
        return self.action

    def single_action_choose(self, action_choice=None):
        if action_choice is None:
            action_choice = self.action_choice
        if action_choice == "Probability":
            if np.all(self.action_neuron_potential == 0):
                self.action_neuron_potential[:] = 1
            action = np.random.choice(self.synapse.number_of_action_neurons, 1,
                                           p=self.action_neuron_potential / np.sum(self.action_neuron_potential))
        elif action_choice == "SoftMax":
            action = np.random.choice(self.synapse.number_of_action_neurons, 1,
                                           p=softmax(
                                               self.action_neuron_potential / np.max(self.action_neuron_potential)))
        elif action_choice == "Max":
            if np.all(self.action_neuron_potential == self.action_neuron_potential[0]):
                if self.action_neuron_potential[0] == 0:
                    self.action_neuron_potential[:] = 1
                action = np.random.choice(self.synapse.number_of_action_neurons, 1,
                                               p=self.action_neuron_potential / np.sum(self.action_neuron_potential))
            else:
                action = [np.argmax(self.action_neuron_potential)]
        return action


    def unsupervised_learning(self, fact_action=None, dt=None, code_max_only=True, first_step=False,
                              if_unexpected_learning=True, normalise_weight=True, learning_rate=None):
        if dt is None:
            dt = self.dt
        if fact_action is None:
            fact_action = self.action
        if learning_rate is None:
            learning_rate = self.learning_rate
            unexpected_learning_rate = self.learning_rate*0.1
        if hasattr(self, 'state_coder_random'):
            silence_index, overactive_index = self.state_coder_random.unsupervised_learning(dt)
            remapped_states = np.append(silence_index, overactive_index)
            if np.any(remapped_states):
                self.connection_matrix[remapped_states, :] = self.default_connection_strength
                self.connection_matrix[:, remapped_states] = self.default_connection_strength
                self.leakage_conductance[remapped_states] = self.default_leakage_conductance
                for state_index in remapped_states:
                    if state_index in self.targets.keys():
                        self.targets.pop(state_index)
                    if state_index in self.obstacle.keys():
                        self.obstacle.pop(state_index)
        connection_update = LearningDirectedNeuronGraph.unsupervised_learning(self, self.state_code, dt, first_step,
                                                                              learning_rate=learning_rate)
        # when there is no state change, connection_update matrix only contains 0. Where there are some states
        # changed, the rest elements in the matrix are 0.
        if code_max_only:
            if self.if_single_action:
                self.synapse.unsupervised_learning(fact_action=onehot(self.synapse.number_of_action_neurons, fact_action),
                                                   first_step=first_step, dt=dt, normalise_weight=normalise_weight,
                                                   learning_rate=learning_rate)
                # self.synapse.check_decrease(mode='')
                if if_unexpected_learning:
                    self.synapse.unsupervised_learning_unexpected(
                        fact_action=onehot(self.synapse.number_of_action_neurons, fact_action),
                        first_step=first_step, dt=dt, code_max_only=code_max_only, route_max_only=None,
                        normalise_weight=normalise_weight, learning_rate=learning_rate)
                    # self.synapse.check_decrease(mode='')


            else:
                self.synapse.unsupervised_learning(fact_action=self.action,
                                                   first_step=first_step, dt=dt, normalise_weight=normalise_weight,
                                                   learning_rate=learning_rate)
                if if_unexpected_learning:
                    self.synapse.unsupervised_learning_unexpected(
                        fact_action=self.action,
                        first_step=first_step, dt=dt, code_max_only=code_max_only, route_max_only=None,
                        normalise_weight=normalise_weight, learning_rate=learning_rate)
        else:
            raise NotImplementedError
        if code_max_only:
            max_index = np.argmax(self.state_code)
            if max_index not in self.targets:
                assert np.all(
                    np.greater_equal(self.leakage_conductance[max_index], 0)), 'leakage conductance less than 0.' + str(
                    self.leakage_conductance[max_index])
                self.leakage_conductance[max_index] = self.default_leakage_conductance + \
                                                      (-self.default_leakage_conductance + self.leakage_conductance[
                                                          max_index]) * np.exp(-dt * 1000) #
                assert np.all(
                    np.greater_equal(self.leakage_conductance[max_index], 0)), 'leakage conductance less than 0.'
        else:
            raise NotImplementedError

    def update_connections_with_reward(self, dt=None, reward=0):
        if dt is None:
            dt = self.dt
        flow_to_matrix = np.multiply(self.current_matrix / self.current_matrix.max(),
                                     np.expand_dims(self.state_code, axis=0))
        self.connection_matrix = positive_variable_update(self.connection_matrix, reward * flow_to_matrix, dt,
                                                          method="reciprocal_saturation")

    def reinforcement_learning(self, dt=None, reward=0, cumu_reward=None, done=False, if_target_state=False,
                               if_danger_state=False, code_max_only=True, first_step=False, fact_action=None, synapse_2nd_RL=True):
        # TODO: first_step
        if dt is None:
            dt = self.dt
        if cumu_reward is None:
            self.cumu_reward += reward
            cumu_reward = self.cumu_reward
        else:
            self.cumu_reward = cumu_reward
        self.reward = reward
        if self.if_single_action:
            fact_action_onehot = np.zeros(self.synapse.number_of_action_neurons)
            fact_action_onehot[fact_action] = 1

        if code_max_only:
            if not first_step and synapse_2nd_RL:
                self.synapse.reinforcement_learning(dt, reward, fact_action_onehot)
            present_state = np.argmax(self.state_code)
            if self.one_target:
                if if_target_state:
                    existing_target_states = list(self.targets.keys())
                    self.set_target(
                        [[an_existing_target_state, None] for an_existing_target_state in existing_target_states])
                    self.set_target([[present_state, 1]])
            else:
                if if_target_state:
                    # self.reset_target()
                    # if present_state in self.targets:
                    if self.targets:
                        if present_state in self.targets:
                            # leakage_conductance = self.targets[present_state] + (1-self.targets[present_state]) * 0.1
                            # self.set_target([[present_state, leakage_conductance]])
                            leakage_conductance_sum = np.sum(list(self.targets.values()))
                            for key, value in self.targets.items():
                                if key == present_state:
                                    leakage_conductance = self.targets[key] + (self.max_ground_conductance - self.targets[key]) * 0.1 * dt
                                else:
                                    leakage_conductance = self.targets[key] + ( 0 -self.targets[key]) * 0.1*dt
                                    # leakage_conductance = self.targets[key] + (
                                    #             self.targets[key] / (leakage_conductance_sum + if_target_state) -
                                    #             self.targets[key]) * 0.1
                                self.set_target([[key, leakage_conductance]])
                        else:
                            self.set_target([[present_state, 1]])

                    else:
                        self.set_target([[present_state, 1]])

                        # self.set_target([[present_state, 100]])
                        # if done:
                        #     if reward > 100 or cumu_reward > 100:
                        #         # inject_current_change_vector = -(reward*1e-7 - self.get_inject_current_vector()) * \
                        #         #                                  softmax(self.potential_vector) * dt * self.target_update_rate
                        #         # inject_current_change_vector = np.zeros(self.number_of_neurons)
                        #         # inject_current_change_vector[np.argmax(self.potential_vector)] = -1e-5
                        #         # self.set_inject_currents_by_vector(inject_current_change_vector)
                        #         present_state = np.argmax(self.state_code)
                        #         self.reset_target()
                        #         self.set_target([[present_state, 1]])
                        # self.connections_matrix_normalise()
                else:
                    present_state = np.argmax(self.state_code)
                    if present_state in self.targets:
                        leakage_conductance = self.targets[present_state] * np.exp(-self.target_decay_rate*dt)
                        self.set_target([[present_state, leakage_conductance]])
            if if_danger_state:
                present_state = np.argmax(self.state_code)
                # self.reset_target()
                self.set_danger([[present_state, 1]])
        else:
            raise NotImplementedError
        if done:
            self.last_cumu_reward = cumu_reward
            self.cumu_reward = 0

    def update(self):
        LearningDirectedNeuronGraph.update(self)
        # AsymmetricalNodalCircuit is updated in self.step.

        self.synapse.update()
        self.state_code_last = self.state_code

    def init_heatmap(self):
        self.heat_vector = np.zeros(self.number_of_neurons)

    def record_heatmap(self, code_max_only=None):
        if code_max_only is None:
            code_max_only = self.code_max_only
        if code_max_only:
            self.heat_vector[np.argmax(self.state_code)] += 1
        else:
            raise NotImplementedError

    def plot_heatmap(self, decoder=None, task=None):
        if task == 'Taxi-v3':
            heat_matrix = np.zeros((5 * 5, 5 * 4))
            for i in range(self.number_of_neurons):
                taxi_row, taxi_col, pass_idx, dest_idx = decoder(i)
                heat_matrix[taxi_row + pass_idx * 5, taxi_col + dest_idx * 5] = self.heat_vector[i]
            fig = heatmap2d(heat_matrix + 1)
        else:
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1])
            ax.bar(range(self.number_of_neurons), self.heat_vector)
        return fig

    # def
    #     self.name_list_log = ['connection_matrix', 'leakage_conductance', 'targets', 'potential_conservations',
    #                           'conservation_vector']
    #
    #     retrieve_record
    # save_as_video(self.retrieve_record(), video_path, self.name)

    def plot_graph(self, set_edge_label=True, *args, **kwargs):
        if set_edge_label:
            edge_label_texts = {(n1, n2): np.array2string(self.synapse.weight[n1, n2, :], precision=3)
                                for n1, n2 in self.for_plot_graph.edges()}
        else:
            edge_label_texts = None
        fig = AsymmetricalNodalCircuit.plot_graph(self, edge_label_texts=edge_label_texts, *args, **kwargs)

        # self.for_plot_edge_index_from_list = edge_index_from_list
        # self.for_plot_edge_index_to_list = edge_index_to_list
        # edge_labels = nx.draw_networkx_edge_labels(self.for_plot_graph, pos=pos,
        #                                            edge_labels=edge_label_texts, ax=ax, font_size=7, alpha=0.8,
        #                                            verticalalignment='center', label_pos=0.3)
        return fig



    def save_as_video(traces, video_path, prefix=''):
        video_filename = os.path.join(video_path, prefix + 'output.avi')
        print('video_filename', video_filename)
        weights = np.sqrt(np.multiply(traces['presynaptic_size'], traces['postsynaptic_size']))
        videoframes = np.concatenate((traces['presynaptic_size'], traces['postsynaptic_size'], weights), axis=2)
        matrix2video2(videoframes, video_filename=video_filename)
        # matrix2video2( traces['postsynaptic_size'], video_filename=video_filename)
        video_filename_pre = os.path.join(video_path, prefix + 'output_pre.avi')
        matrix2video2(traces['presynaptic_size'], video_filename=video_filename_pre)
        video_filename_post = os.path.join(video_path, prefix + 'output_post.avi')
        matrix2video2(traces['postsynaptic_size'], video_filename=video_filename_post)
        video_filename_weight = os.path.join(video_path, prefix + 'output_weight.avi')
        matrix2video2(weights, video_filename=video_filename_weight)


# if map_only:
#     self.generate_graph(number_of_edges=625)
# else:
#     self.generate_graph(number_of_edges=25*4*4*5)
# self.plot_graph()
# self.plot_heatmap(decoder=env.decode, task=environment)
# self.save_graph(path_dict['model'])
# import networkx as nx
# import os
#
# nx.readwrite.edgelist.write_edgelist(self.graph, path=os.path.join(path_dict['model_path'], 'graph.csv'))
# numpy.unravel_index(self.current_matrix.argmax(), self.current_matrix.shape)
if __name__ == "__main__":
    number_of_states = 10
    number_of_inputs = 8
    DRN = DynamicRoutingNet(number_of_states)
    DRN.init_full_random_mapping(number_of_inputs=number_of_inputs)
    DRN.init_second_order_synapse(number_of_action_neurons=4)
    DRN.step(np.random.random(number_of_inputs))
    DRN.unsupervised_learning()
    DRN.reinforcement_learning(reward=1)
    DRN.update()
    DRN.generate_graph()
    DRN.plot_graph()
    plt.show()
