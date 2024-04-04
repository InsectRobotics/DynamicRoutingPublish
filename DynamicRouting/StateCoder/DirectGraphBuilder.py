import copy

import dill
import matplotlib.pyplot as plt
import numpy as np
from DynamicRouting.Core.DirectedNeuronGraph import DirectedNeuronGraph
from DynamicRouting.StateCoder.FullRandomMapping import FullRandomMapping
from DynamicRouting.Utils.pathselection import common_path
from DynamicRouting.Utils.plot import save_plots_as_pdf


class LearningDirectedNeuronGraph(DirectedNeuronGraph):
    """

    """
    def __init__(self, number_of_neurons, default_connection_strength, max_connection_strength=10,
                 learning_rate=0.05, dt=0.01,
                 increase_threshold=0, decrease_threshold=0, code_max_only=True):
        super().__init__(number_of_neurons, default_connection_strength)
        self.learning_rate = learning_rate
        self.dt = dt
        self.increase_threshold = increase_threshold
        self.decrease_threshold = decrease_threshold
        self.neuron_activates = np.zeros(number_of_neurons)
        self.neuron_activates_last = np.zeros(number_of_neurons)
        self.code_max_only = code_max_only
        self.max_connection_strength = max_connection_strength

    def unsupervised_learning(self, neuron_activates, dt=None, first_step=False, learning_rate=None):
        if dt is None:
            dt = self.dt
        if learning_rate is None:
            learning_rate = self.learning_rate
        if self.code_max_only:
            max_index = np.argmax(neuron_activates)
            self.neuron_activates[:] = 0
            self.neuron_activates[max_index] = neuron_activates[max_index]
        else:
            self.neuron_activates = neuron_activates
        if not first_step:  # avoid learning state from last episode
            self.active_diff = self.neuron_activates - self.neuron_activates_last
            self.active_diff_rate = self.active_diff / dt
            self.active_diff_cov = np.outer(self.active_diff_rate, self.active_diff_rate)
            self.increaseing_neuron = np.greater(self.active_diff_rate, self.increase_threshold)
            self.decreaseing_neuron = np.less(self.active_diff_rate, self.decrease_threshold)
            self.valid_update = np.outer(self.decreaseing_neuron, self.increaseing_neuron)
            np.fill_diagonal(self.valid_update, False)
            self.connection_update = -np.multiply(self.active_diff_cov, self.valid_update)* learning_rate
            # For a connection that is building up, its upstream neuron's activity is decreasing and downstream neuron's
            # activity is increasing, so the covariance of changes is negative.
            # [last, current] = np.unravel_index(np.argmax(self.connection_update), self.connection_update.shape)
            # [last_row, last_col, last_pass, last_dest] = decode(last)
            # [current_row, current_col, current_pass, current_dest] = decode(current)
            # assert abs(current_row-last_row)+abs(current_col-last_col) <= 1, 'The state jumped.'
            # self.update_connections_with_matrix(self.connection_update * dt)

            # self.connection_matrix += (self.max_connection_strength-self.connection_matrix)*self.connection_update*dt

            # postive_index = self.connection_update>0
            # self.connection_matrix[postive_index] += (self.max_connection_strength - self.connection_matrix[postive_index]) \
            #                                             * self.connection_update[postive_index]*dt
            # negative_index = self.connection_update<0
            # self.connection_matrix[negative_index] += - self.connection_matrix[negative_index]\
            #                                             * np.exself.connection_update[negative_index]*dt
            self.connection_matrix += (self.max_connection_strength - self.connection_matrix) * self.connection_update * dt
            self.connection_matrix[self.connection_matrix < 0] = 0
            self.connection_matrix[self.connection_matrix > self.max_connection_strength] = self.max_connection_strength
        else:
            self.connection_update = np.zeros([self.number_of_neurons, self.number_of_neurons])
        return self.connection_update

    def update(self):
        self.neuron_activates_last = copy.deepcopy(self.neuron_activates)



if __name__ == "__main__":
    DEBUG = 1
    # Parameters
    experiment = 'DirectedGraphState'
    code_max_only = True
    path_dict = common_path(experiment)
    figure_dict = dict()
    figure_index = 0
    BWD_trace_path = '../../../DynamicSynapse/Data/BipedalWalker-v3/BipedalWalkerDemoTrace.pkl'
    with open(BWD_trace_path, mode='rb') as fl:
        trace = dill.load(fl)
    observations = np.array(trace['observation'])
    dt = 0.02
    number_of_steps = 10000
    number_of_inputs = observations.shape[1]
    number_of_states = 10
    FRM = FullRandomMapping(number_of_inputs, number_of_states, dt=dt, threshold=5,
                            activity_check_start_time=4, code_max_only=code_max_only,
                            silence_count_threshold=0.02, overactive_count_threshold=0.3)
    default_connection_strength = 1e-4
    KCs = LearningDirectedNeuronGraph(number_of_states, default_connection_strength, code_max_only=code_max_only)

    maxes = np.zeros(number_of_steps, dtype=int)

    for index_of_step in range(number_of_steps):
        output_value = FRM.step(observations[index_of_step, :])
        maxes[index_of_step] = np.argmax(output_value)
        KCs.update()
        KCs.unsupervised_learning(output_value)
        FRM.unsupervised_learning()
        if DEBUG:
            print('index_of_step', index_of_step)
            print('silence_index', FRM.silence_index)
            print('overactive_index', FRM.overactive_index)
            print('index of max output_value: ', maxes[index_of_step])
            # print('connection_matrix \n', KCs.connection_matrix)
    plt.plot(maxes)
    print('Average activity.', FRM.activity_counter/FRM.time_counter)

    KCs.generate_graph()
    figure_dict[figure_index] = KCs.plot_graph(title="learned Directed Neuron Graph")
    save_plots_as_pdf(figure_dict, path_dict['plot_path'], file_name='plots.pdf')
    plt.show(block=False)
    pass