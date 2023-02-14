import copy

import numpy as np
from DynamicRouting.Core.DirectedNeuronGraph import DirectedNeuronGraph


class LearningDirectedNeuronGraph(DirectedNeuronGraph):

    def __init__(self, number_of_neurons, default_connection_strength, max_connection_strength=10,
                 learning_rate=1e-3, dt=0.01,
                 increase_threshold=0, decrease_threshold=0):
        super().__init__(number_of_neurons, default_connection_strength)
        self.learning_rate = learning_rate
        self.dt = dt
        self.increase_threshold = increase_threshold
        self.decrease_threshold = decrease_threshold
        self.neuron_activates = np.zeros(number_of_neurons)
        self.neuron_activates_last = np.zeros(number_of_neurons)
        self.max_connection_strength = max_connection_strength

    def unsupervised_learning(self, neuron_activates, dt=None, first_step=False):
        if dt is None:
            dt = self.dt
        max_index = np.argmax(neuron_activates)
        self.neuron_activates[:] = 0
        self.neuron_activates[max_index] = neuron_activates[max_index]
        if not first_step:  # avoid learning state from last episode
            self.active_diff = self.neuron_activates - self.neuron_activates_last
            self.active_diff_rate = self.active_diff / dt
            self.active_diff_cov = np.outer(self.active_diff_rate, self.active_diff_rate)
            self.increaseing_neuron = np.greater(self.active_diff_rate, self.increase_threshold)
            self.decreaseing_neuron = np.less(self.active_diff_rate, self.decrease_threshold)
            self.valid_update = np.outer(self.decreaseing_neuron, self.increaseing_neuron)
            np.fill_diagonal(self.valid_update, False)
            self.connection_update = -np.multiply(self.active_diff_cov, self.valid_update)*self.learning_rate
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

