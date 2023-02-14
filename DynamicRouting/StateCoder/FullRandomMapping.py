import dill
import numpy as np
import matplotlib.pyplot as plt
from DynamicRouting.Utils.MathUtil import relu

class FullRandomMapping:
    def __init__(self, Number_of_inputs, Number_of_outputs, distribution=np.random.normal, threshold=1,
                 silence_count_threshold=0.01, overactive_count_threshold=0.5,
                 dt=0.01, activity_check_start_time=100, learning_rule='remap',
                 factor_update_rate=1e-2):
        self.Number_of_inputs = Number_of_inputs
        self.Number_of_outputs = Number_of_outputs
        self.distribution = distribution
        self.factor = self.distribution(size=(Number_of_inputs, Number_of_outputs))
        self.activity_counter = np.zeros(self.Number_of_outputs)
        self.time_counter = np.zeros(self.Number_of_outputs)
        self.threshold = threshold
        self.silence_count_threshold = silence_count_threshold
        self.overactive_count_threshold = overactive_count_threshold
        self.dt = dt
        self.activity_check_start_time = activity_check_start_time
        self.silence_index = []
        self.overactive_index = []
        self.output_value = np.zeros((Number_of_inputs, Number_of_outputs))
        self.necessity_to_check = np.ones((Number_of_outputs))
        self.learning_rule = learning_rule
        self.factor_update_rate = factor_update_rate

    def remap(self, indexes_of_outputs):
        indexes_of_outputs = np.array(indexes_of_outputs)
        if indexes_of_outputs.size > 0:
            self.factor[:, indexes_of_outputs] = self.distribution(size=(self.Number_of_inputs, len(indexes_of_outputs)))
            self.time_counter[indexes_of_outputs] = 0
            self.activity_counter[indexes_of_outputs] = 0

    def rescale_continousely(self, dt):
        if dt is None:
            dt = self.dt
        if len(self.silence_index) > 0:
            self.factor[:, self.silence_index] += dt * self.factor_update_rate * self.factor[:, self.silence_index] * \
                                                  (self.silence_count_threshold \
                                                - self.activity_counter[self.silence_index] / self.time_counter[self.silence_index])
        if len(self.overactive_index) > 0:
            self.factor[:, self.overactive_index] += dt * self.factor_update_rate * self.factor[:, self.overactive_index] * \
                                                     (self.overactive_count_threshold \
                                                    - self.activity_counter[self.overactive_index] / self.time_counter[self.overactive_index])

    def rescale_once(self):
        if len(self.silence_index) > 0:
            zero_activity_index = np.equal(self.activity_counter[self.silence_index], 0)
            non_zero_activity_index = np.greater(self.activity_counter[self.silence_index], 0)
            self.factor[:, self.silence_index][:, non_zero_activity_index] = \
                self.factor[:, self.silence_index][:, non_zero_activity_index] * \
                (self.silence_count_threshold \
                / (self.activity_counter[self.silence_index][non_zero_activity_index]
                   / self.time_counter[self.silence_index][non_zero_activity_index]))
            # self.factor[:, self.silence_index][:, zero_activity_index] = \
            #     self.factor[:, self.silence_index][:, zero_activity_index] * 1.1
            self.remap(self.silence_index[zero_activity_index])
        if len(self.overactive_index) > 0:
            self.factor[:, self.overactive_index] = self.factor[:, self.overactive_index] * \
                                                     (self.overactive_count_threshold \
                                                    / (self.activity_counter[self.overactive_index] / self.time_counter[self.overactive_index]))

    def activity_check(self, dt):
        if dt is None:
            dt = self.dt
        self.time_counter += dt
        max_index = np.argmax(self.output_value)
        self.activity_counter[max_index] += dt
        self.necessity_to_check_by_time = self.necessity_to_check/self.activity_counter
        self.state_can_be_checked = np.greater(self.time_counter, self.activity_check_start_time)
        self.silence_index = np.where(np.logical_and(self.state_can_be_checked,
            np.less(self.activity_counter, self.silence_count_threshold * self.time_counter)))[0]
        self.overactive_index = np.where(np.logical_and(self.state_can_be_checked,
            np.greater(self.activity_counter, self.overactive_count_threshold * self.time_counter)))[0]
        return self.silence_index, self.overactive_index

    def step(self, input_value, dt=None, update_mapping=False):
        if dt is None:
            dt = self.dt
        self.input_value = input_value
        if update_mapping:
            if self.learning_rule == 'remap':
                self.remap(np.append(self.silence_index, self.overactive_index))
            elif self.learning_rule == 'rescale':
                self.rescale_once()
            else:
                raise NotImplementedError
        self.output_value = relu(np.matmul(self.factor.T, self.input_value))
        if update_mapping:
            self.activity_check(dt)
        return self.output_value

    def unsupervised_learning(self, dt=None):
        if dt is None:
            dt = self.dt
        if self.learning_rule == 'remap':
            self.remap(np.append(self.silence_index, self.overactive_index))
        elif self.learning_rule == 'rescale':
            self.rescale_once()
        else:
            raise NotImplementedError
        self.activity_check(dt)
        return [self.silence_index, self.overactive_index]

    def reinfrocement_learning(self, dt=None, reward=0):
        if dt is None:
            dt = self.dt
        if reward > 0:
            index_of_max = np.argmax(self.output_value)
            self.factor[:, index_of_max] += \
                (self.input_value - self.factor[:, index_of_max]) * dt * reward
            self.necessity_to_check[index_of_max] = index_of_max * np.exp(-reward * dt)


if __name__ == '__main__':
    BWD_trace_path = '../../../DynamicSynapse/Data/BipedalWalker-v3/BipedalWalkerDemoTrace.pkl'
    with open(BWD_trace_path, mode='rb') as fl:
        trace = dill.load(fl)
    observations = np.array(trace['observation'])
    dt = 0.02
    number_of_steps = 10000
    number_of_inputs = observations.shape[1]
    number_of_outputs = 10
    FRM = FullRandomMapping(number_of_inputs, number_of_outputs, dt=dt, threshold=5,
                            activity_check_start_time=4)
    maxes = np.zeros(number_of_steps)
    for index_of_step in range(number_of_steps):
        output_value = FRM.step(observations[index_of_step, :])
        FRM.unsupervised_learning()
        print('index_of_step', index_of_step)
        print('silence_index', FRM.silence_index)
        print('overactive_index', FRM.overactive_index)
        maxes[index_of_step] = np.argmax(output_value)
        print('index of max output_value: ', maxes[index_of_step])
    plt.plot(maxes)
    print('Average activity.', FRM.activity_counter / FRM.time_counter)
    plt.show()
    pass
