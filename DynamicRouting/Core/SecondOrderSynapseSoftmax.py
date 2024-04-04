import copy

import numpy as np
from DynamicSynapse.Utils.loggable import Loggable

from DynamicRouting.Utils.MathUtil import softmax, softmaxM1, normalised_log_0, positive_variable_update, relu, \
    rational_variable_update, sigmoid
from DynamicRouting.Utils.MatrixAndOrders import matrix2DXmatrix3D, matrix2D_mul_matrix3D

# import numba as nb
#
#
# @np.njit
# def valid_highest_potential(state_neruons_potential, presynaptic_current):
#     order = np.argsort(state_neruons_potential)
#     for index in range(len(order)):
#         max_current_i = np.argmax(presynaptic_current[order[-index]])
#         if max_current_i < 1


class SecondOrderSynapsSoftmax(Loggable):
    """
    This implementation considers the effect of presynaptic synapse with lateral inhibition approximated by softmax.
    Referring to the lateral inhibition version for implementation without max function, which is more biologically
    plausible.
    """

    def __init__(self, number_of_state_neurons, number_of_action_neurons, weight=None, preference='', dt=0.02,
                 increase_threshold=0, decrease_threshold=0, code_max_only=True, route_max_only=True,
                 if_single_action=True, guider_type='power', learning_rate=0.05):
        # learning_rate was 1
        self.number_of_state_neurons = number_of_state_neurons
        self.number_of_action_neurons = number_of_action_neurons
        if weight is None:
            self.weight = np.random.random(
                (self.number_of_state_neurons, self.number_of_state_neurons, self.number_of_action_neurons))
        elif np.isscalar(weight):
            self.weight = np.ones(
                (self.number_of_state_neurons, self.number_of_state_neurons, self.number_of_action_neurons))* weight
        else:
            assert weight.shape == (
                self.number_of_state_neurons, self.number_of_state_neurons, self.number_of_action_neurons)
            self.weight = weight
        self.action_neuron_current = np.zeros(self.number_of_action_neurons)
        self.preference = preference
        self.weight_change = np.zeros(self.weight.shape)
        self.dt = dt
        self.action_neuron_posynaptic_current = \
            np.zeros((self.number_of_state_neurons, self.number_of_state_neurons, self.number_of_action_neurons))
        self.state_neuron_potentials = np.zeros(self.number_of_state_neurons)
        self.state_neuron_potentials_last = copy.deepcopy(self.state_neuron_potentials)
        self.increase_threshold = increase_threshold
        self.decrease_threshold = decrease_threshold
        self.code_max_only = code_max_only
        self.route_max_only = route_max_only
        self.guider_type = guider_type
        self.guider = np.zeros((self.number_of_state_neurons,self.number_of_state_neurons))
        self.guider_last = copy.deepcopy(self.guider)
        self.state_neuron_potentials = np.zeros(self.number_of_state_neurons)
        self.inter_state_neuron_current = np.zeros((self.number_of_state_neurons, self.number_of_state_neurons))
        self.max_route_index = 0
        self.max_route_index_last = 0
        self.actual_state_index_present = 0
        self.actual_state_index_last = 0
        self.state_code = np.zeros(self.number_of_state_neurons)
        self.state_code_last = copy.deepcopy(self.state_code)
        self.if_single_action = if_single_action
        self.learning_rate = learning_rate
        self.name_list_log = ['weight', 'actual_state_index_present', 'guider', 'max_route_index', 'action_neuron_posynaptic_current',
                              'action_neuron_current']


    def step(self, state_neruon_potentials, inter_state_neuron_current, code_max_only=None, route_max_only=None, guider_type=None,
             state_code=None, route_exploration=False, guider_noise_variance=0):
        '''

        :param state_neruons: a vector of potentials of presynaptic_synapses, which is equivalent to the poential of the
            presynaptic neurons of the presynaptic synapses
        :param inter_state_neuron_current: a matrix of the current of presynaptic synapses.
        :return: The postsynaptic current to the postsynaptic neurons
        '''
        if guider_type is None:
            guider_type = self.guider_type
        # else:
        #     guider_type = 'power'
        # TODO distinguishi guider_type and guider
        self.state_neuron_potentials = state_neruon_potentials
        self.inter_state_neuron_current = inter_state_neuron_current
        if code_max_only is None:
            code_max_only = self.code_max_only
        if route_max_only is None:
            route_max_only = self.route_max_only
        if code_max_only:
            if state_code is not None:
                # TODO modify the continuous case the match the change
                self.actual_state_index_present = np.argmax(state_code)
                self.state_code = state_code
            else:
                self.actual_state_index_present = np.argmax(self.state_neuron_potentials)
            self.action_neuron_posynaptic_current[:, :, :] = 0
            potential_diff = (self.state_neuron_potentials[self.actual_state_index_present] - self.state_neuron_potentials)
            if guider_type == "power":
                guider = relu(self.inter_state_neuron_current[self.actual_state_index_present, :] * potential_diff)
                #TODO: Relu is a temp fix to avoid nagetive guider, which should not have happend.
            elif guider_type == 'current':
                guider = self.inter_state_neuron_current[self.actual_state_index_present, :]
            elif guider_type == 'sum':
                guider = self.inter_state_neuron_current[self.actual_state_index_present, :] \
                        + relu(self.state_neuron_potentials[self.actual_state_index_present] - self.state_neuron_potentials)
            else:
                NotImplementedError
            if route_max_only:
                # if self.actual_state_index_last == 0:
                #     if np.all(self.state_neuron_potentials_last == self.state_neuron_potentials_last[0]):
                #         return
                if route_exploration:
                    if guider_noise_variance == 0:
                        if sum(guider) == 0 and guider[0] == 0:
                            self.max_route_index = np.random.choice(self.number_of_state_neurons, 1,
                                                                    p=np.ones_like(guider)/guider.size)[0]
                        else:
                            self.max_route_index = np.random.choice(self.number_of_state_neurons, 1,
                                                            p=softmax(guider / guider.max()*10))[0]
                    else:
                        guider += sigmoid(np.random.normal(np.zeros_like(guider), guider_noise_variance))
                        self.max_route_index = np.random.choice(self.number_of_state_neurons, 1,
                                                                p=guider/sum(guider))[0]
                else:
                    if np.all(np.abs(guider-guider[0]) < 1e-10): # was 1e-32 before20230910
                        self.max_route_index = np.random.choice(guider.shape[0], 1)[0]
                    else:
                        self.max_route_index = np.argmax(guider)
                self.action_neuron_posynaptic_current[self.actual_state_index_present, self.max_route_index, :] = \
                    self.weight[self.actual_state_index_present, self.max_route_index]
                self.action_neuron_current = self.action_neuron_posynaptic_current[self.actual_state_index_present, self.max_route_index, :]
            else:
                self.action_neuron_posynaptic_current[self.actual_state_index_present, :, :] = \
                    np.multiply(guider, self.weight[self.actual_state_index_present, :, :].T).T
                self.action_neuron_current = self.action_neuron_posynaptic_current.sum(axis=0).sum(axis=0)
        else:
            raise NotImplementedError

        return self.action_neuron_current

    def check_decrease(self, mode='New', action_to_check=3):
        if mode == 'New':
            self.weight_p = copy.deepcopy(self.weight)
        else:
            decreased_weight = np.less(self.weight, self.weight_p)
            if np.any(decreased_weight) and self.max_action_index==action_to_check and self.actual_state_index_last==2 and self.max_route_index_last==4:
                decreased_weight_index = np.where(decreased_weight)
                print(decreased_weight_index)
                return decreased_weight_index
        return None

    def unsupervised_learning_unexpected(self, dt=None, fact_action=0, code_max_only=None, route_max_only=None,
                                         first_step=False, guide=None, normalise_weight=True, learning_rate=None):
        if dt is None:
            dt = self.dt
        if code_max_only is None:
            code_max_only = self.code_max_only
        if route_max_only is None:
            route_max_only = self.route_max_only
        if learning_rate is None:
            learning_rate = self.learning_rate
        # if state_code is not None:
        #     # TODO modify the continuous case the match the change
        #     self.actual_state_index_present = np.argmax(state_code)
        #     self.state_code = state_code
        # else:
        #     self.actual_state_index_present = np.argmax(self.state_neuron_potentials)
        if not first_step:
            if code_max_only and route_max_only:
                if self.if_single_action:
                    # if self.actual_state_index_last == 0:
                    #     if np.all(self.state_neuron_potentials_last == self.state_neuron_potentials_last[0]):
                    #         return
                    self.max_action_index = np.argmax(fact_action)
                    if self.actual_state_index_present != self.max_route_index_last:
                        learning_factor = 400
                        self.weight[self.actual_state_index_last, self.max_route_index_last,  self.max_action_index] \
                            = rational_variable_update(
                            self.weight[self.actual_state_index_last, self.max_route_index_last,  self.max_action_index],
                            0, dt * fact_action[self.max_action_index] * learning_factor * learning_rate, method="proportional") # TODO only learning_rate with 1

                        # self.check_decrease(mode='')

                        if self.weight[self.actual_state_index_last, self.max_route_index_last,  self.max_action_index]<0:
                            print("self.weight[self.actual_state_index_last, self.max_route_index_last,  self.max_action_index]<0, it is %f now"
                                  %(self.weight[self.actual_state_index_last, self.max_route_index_last,  self.max_action_index]))

                            # self.check_decrease(mode='')

                        if normalise_weight:
                            self.weight[self.actual_state_index_last, self.max_route_index_last] /= np.sum(
                                self.weight[self.actual_state_index_last, self.max_route_index_last, :])

                            # self.check_decrease(mode='')

                else:
                    if self.actual_state_index_present != self.max_route_index_last:
                        self.weight[self.actual_state_index_last, self.max_route_index_last, :] \
                            = rational_variable_update(
                            self.weight[self.actual_state_index_last, self.max_route_index_last, :],
                            fact_action, -dt * 20 * learning_rate, method="proportional")
                        if normalise_weight:
                            self.weight[self.actual_state_index_last, self.max_route_index_last] /= np.sum(
                                self.weight[self.actual_state_index_last, self.max_route_index_last, :])

            else:
                self.guider_last_sum = self.guider_last.sum(axis=0)
                self.unexpection = self.state_neuron_potentials - self.guider_last_sum/self.guider_last_sum.max

                self.weight += matrix2D_mul_matrix3D(np.multiply(self.unexpection, self.guider_last),
                                                     (self.weight - fact_action), axis_to_keep_in_B=3) * dt * learning_rate



    def unsupervised_learning(self, dt=None, fact_action=0, code_max_only=True,
                              first_step=False, connection_update=None, normalise_weight=True, learning_rate=None ):
        if learning_rate is None:
            learning_rate = self.learning_rate
        assert not np.any(np.less(fact_action, 0)), "Negative action_neuron_potential"
        if dt is None:
            dt = self.dt
        # self.weight_change = state_connection_update
        # for index in range(self.number_of_action_neurons):
        #     self.weight_change[:, :, index] = state_connection_update * self.action_neuron_current[index]
        # self.weight = positive_variable_update(self.weight, self.weight_change, dt,
        #                                        method="reciprocal_saturation")
        # weight_change_target = np.zeros_like(self.weight)
        # for index in range(self.number_of_action_neurons):
        #     weight_change_target[:, :, index] = synapse_update_target
        # self.weight = rational_variable_update(self.weight, weight_change_target, dt * action_neuron_potential,
        #                                        method="proportional")

        if code_max_only:
            if not first_step:
                self.actual_state_index_present = np.argmax(self.state_code)
                self.actual_state_index_last = np.argmax(self.state_code_last)
                if self.actual_state_index_last != self.actual_state_index_present:
                    self.max_action_index = np.argmax(fact_action)
                    # self.weight[self.actual_state_index_last, self.actual_state_index_present, self.max_action_index] += 0.1*dt
                    self.weight[self.actual_state_index_last, self.actual_state_index_present, self.max_action_index] += learning_rate *dt*2 #0.05*dt
        else:
            if not first_step:  # avoid learning state from last episode
                if connection_update is None:
                    self.active_diff = self.state_code.astype(float) - self.state_code_last.astype(float)
                    self.active_diff_rate = self.active_diff / dt
                    self.active_diff_cov = np.outer(self.active_diff_rate, self.active_diff_rate)
                    self.increaseing_neuron = np.greater(self.active_diff_rate, self.increase_threshold)
                    self.decreaseing_neuron = np.less(self.active_diff_rate, self.decrease_threshold)
                    self.valid_update = np.outer(self.decreaseing_neuron, self.increaseing_neuron)
                    np.fill_diagonal(self.valid_update, False)
                    connection_update = np.sqrt(-np.multiply(self.active_diff_cov, self.valid_update))
                    assert not np.any(np.less(connection_update, 0)), "Negative or zero connection_update"
                connection_update = connection_update * learning_rate  # TODO arbitrary factor
                # For a connection that is building up, its upstream neuron's activity is decreasing and downstream neuron's
                # activity is increasing, so the covariance of changes is negative.
                self.weight_change_target = np.zeros_like(self.weight)
                for index in range(self.number_of_action_neurons):
                    self.weight_change_target[:, :, index] = connection_update * fact_action # the code can be optimised considering fact_action
            else:
                self.weight_change_target = copy.deepcopy(self.weight)
                # self.update()
            # self.weight = rational_variable_update(self.weight, self.weight_change_target, dt* fact_action, # use fact_action array to allow update excuted action
            #                                        method="proportional")
            self.weight = self.weight+self.weight_change_target*dt * 50 # TODO parameterise 50; check if fact_action is one_hot. It should be.

        if normalise_weight:
            self.weight /= np.sum(self.weight, axis=2, keepdims=True)

        # [last, current, action_ind] = np.unravel_index(np.argmax(self.weight_change_target), self.weight_change_target.shape)
        # self.weight += self.weight_change
        # self.weight /= self.weight.sum()
        return

    def reinforcement_learning(self, dt=None, reward=0, fact_action=0, code_max_only=True, learning_rate=None):
        if dt is None:
            dt = self.dt
        if code_max_only is None:
            code_max_only = self.code_max_only
        if learning_rate is None:
            learning_rate = self.learning_rate
        if code_max_only:
            max_action_index = np.argmax(fact_action)
            max_current_index_1D = np.argmax(self.action_neuron_posynaptic_current[:, :, max_action_index])
            max_current_index = np.unravel_index(max_current_index_1D,
                                                 (self.number_of_state_neurons, self.number_of_state_neurons))
            # synapse_weight_change_matrix[max_current_index[0], max_current_index[1], max_action_index] += 1
            # TODO check the order of max_current_index[1], max_current_index[0],
            self.weight[max_current_index[1], max_current_index[0], max_action_index] = \
                rational_variable_update(
                    self.weight[max_current_index[1], max_current_index[0], max_action_index],
                    1 * np.sign(reward), dt * np.abs(reward)*learning_rate, method="proportional") #TODO
                     # 1 * np.sign(reward), dt * np.abs(reward), method = "proportional")  # TODO
        else:
            raise NotImplementedError

    def update(self):
        self.state_neuron_potentials_last = copy.deepcopy(self.state_neuron_potentials)
        if self.route_max_only:
            self.max_route_index_last = self.max_route_index
        self.actual_state_index_last = self.actual_state_index_present
        self.state_code_last = self.state_code
        self.guider_last = self.guider

if __name__ == "__main__":
    number_of_state_neruons = 10
    number_of_action_neurons = 5
    mods = np.arange(number_of_state_neruons) % number_of_action_neurons
    weight = np.zeros((number_of_state_neruons, number_of_state_neruons, number_of_action_neurons))
    weight[np.arange(number_of_state_neruons), :, mods] = 1
    SOSO = SecondOrderSynapsSoftmax(number_of_state_neruons, number_of_action_neurons, weight)
    presynaptic_potential = np.linspace(0, 1, 10)
    presynaptic_current = np.random.random((10, 10))
    action_neuron_current = SOSO.step(presynaptic_potential, presynaptic_current)
    print(action_neuron_current)
    weight_change = np.zeros((number_of_state_neruons, number_of_state_neruons))
    weight_change[0, 0] = 1
    SOSO.unsupervised_learning(weight_change * 0.1, learning_rate=1e-4)
    print(SOSO.weight)
    pass
