import numpy as np

from DynamicRouting.Core.AsymmetricalNodalCircuit import AsymmetricalNodalCircuit
from DynamicRouting.Utils.MatrixAndOrders import expand_vector, expand_matrix
from DynamicRouting.Utils.pathselection import common_path
from DynamicRouting.Utils.plot import save_plots_as_pdf
import warnings
import torch
import numba as nb


@nb.njit
def index_remove(conservation_vector, nodal_matrix):
    index_to_remove = []
    for index in range(conservation_vector.shape[0]):
        if np.all(nodal_matrix[index, :] == 0) and np.all(nodal_matrix[:, index] == 0):
            index_to_remove.append(index)
    index_connected = np.delete(np.arange(conservation_vector.shape[0]), index_to_remove)
    return index_to_remove, index_connected

class AsymmetricalNodalCircuitExtraGround(AsymmetricalNodalCircuit):
    """

    """

    def __init__(self, number_of_neurons, default_connection_strength, default_leakage_conductance=1e-10):
        self.leakage_conductance = np.full(number_of_neurons, default_leakage_conductance)
        self.default_leakage_conductance = default_leakage_conductance
        super().__init__(number_of_neurons, default_connection_strength)
        np.fill_diagonal(self.nodal_admittance_matrix,
                         self.nodal_admittance_matrix.diagonal() + self.leakage_conductance)
        self.leakage_current = np.zeros(self.number_of_neurons)
        self.targets = dict()
        self.dangers = dict()

    def update_nodal_matrix(self):
        self.nodal_admittance_matrix[self.passable_connection] \
            = -self.connection_matrix[self.passable_connection]
        self.nodal_admittance_matrix[self.passable_connection.T] \
            = self.nodal_admittance_matrix.T[self.passable_connection.T]
        np.fill_diagonal(self.nodal_admittance_matrix, 0)
        np.fill_diagonal(self.nodal_admittance_matrix,
                         -self.nodal_admittance_matrix.sum(axis=0) + self.leakage_conductance)
        self.nodal_matrix[:self.number_of_neurons, :self.number_of_neurons] = self.nodal_admittance_matrix

    def _add_potential_conservation(self, i, j, k, conservation):
        self.conservation_vector[k] = conservation
        if i != -1:
            self.nodal_matrix[i, k] = 1
            self.nodal_matrix[k, i] = 1
        if j != -1:
            self.nodal_matrix[j, k] = -1
            self.nodal_matrix[k, j] = -1

    def add_potential_conservations(self, conservations, update_existing_state=True):
        self.conservation_vector = expand_vector(self.conservation_vector, len(conservations))
        self.nodal_matrix = expand_matrix(self.nodal_matrix, len(conservations))
        for i, j, conservation in conservations:
            if update_existing_state and i != -1 and j != -1:
                conservation += self.potential_diff_matrix[i, j]
            self._add_potential_conservation(i, j,
                                             self.number_of_potential_conservations + self.number_of_neurons,
                                             conservation)
            if not i in self.potential_conservations:
                self.potential_conservations[i] = dict()
            else:
                assert not j in self.potential_conservations[i], \
                    "The conservation conflicts with an existing conservation."
            self.potential_conservations[i][j] = {"value": conservation,
                                                  "index": self.number_of_potential_conservations + self.number_of_neurons}
            self.number_of_potential_conservations += 1

    def reset_potential_conservations(self):
        remove_list = []
        for i in self.potential_conservations.keys():
            for j in self.potential_conservations[i].keys():
                remove_list.append([i, j, None])
        self.update_potential_conservations(remove_list)

    def update_current(self):
        super().update_current()
        self.leakage_current = np.multiply(self.potential_vector, self.leakage_conductance)

    def set_ground(self, *args, **kwargs):
        warnings.warn("Ground has been set outside of the neurons, connecting to every neurons with an average leakage "
                      "resistance {:e}.".format(self.leakage_conductance.mean()))

    def reset_target(self):
        for i, conductance in self.targets.items():
            self.leakage_conductance[i] -= conductance
            assert np.all(
                np.greater_equal(self.leakage_conductance[i], 0)), 'leakage conductance less than 0. Index' + str(i)
        self.targets.clear()

    def set_danger(self, dangers):
        for i, danger_rate in dangers:
            if i in self.dangers:
                if danger_rate is None:
                    self.connection_matrix[:, i] *= self.dangers[i]
                    self.dangers.pop(i)
            else:
                self.dangers[i] = danger_rate
                self.leakage_conductance[i] /= danger_rate

    def set_target(self, targets):
        for i, conductance in targets:
            assert np.all(np.greater_equal(self.leakage_conductance[i], 0)), 'leakage conductance less than 0.'
            if i in self.targets:
                if conductance is None:
                    self.leakage_conductance[i] -= self.targets[i]
                    self.targets.pop(i)
                else:
                    self.leakage_conductance[i] -= self.targets[i]
                    self.targets[i] = conductance
                    self.leakage_conductance[i] += conductance
            else:
                self.targets[i] = conductance
                self.leakage_conductance[i] += conductance
            assert np.all(np.greater_equal(self.leakage_conductance[i], 0)), 'leakage conductance less than 0.'

    def update_target_with_vector(self, targe_change_vector):
        self.leakage_conductance += targe_change_vector
        assert np.all(np.greater(self.leakage_conductance, 0)), 'leakage conductance less than 0.'

    def set_target_with_vector(self, targe_change_vector):
        self.leakage_conductance = targe_change_vector
        assert np.all(np.greater_equal(self.leakage_conductance, 0)), 'leakage conductance less than 0.'

    def solve(self):
        # self.solves = np.linalg.solve(self.nodal_matrix, self.conservation_vector)
        try:
            self.solves = torch.linalg.solve(torch.from_numpy(self.nodal_matrix),
                                             torch.from_numpy(self.conservation_vector)).numpy()
        except:
            index_to_remove, self.index_connected = index_remove(self.conservation_vector, self.nodal_matrix)
            self.solves = np.zeros(self.conservation_vector.shape[0])
            try:
                self.solves[self.index_connected] = torch.linalg.solve(
                    torch.from_numpy(self.nodal_matrix[self.index_connected, :][:, self.index_connected]),
                    torch.from_numpy(self.conservation_vector[self.index_connected])).numpy()
            except:
                self.solves[self.index_connected] = np.linalg.lstsq(self.nodal_matrix[self.index_connected, :][:, self.index_connected],
                                              self.conservation_vector[self.index_connected])[0]
        self.potential_vector = self.solves[:self.number_of_neurons]
        return self.solves



if __name__ == "__main__":
    # Parameters
    experiment = 'AsymmetricalNodalCircuitExtraGround'
    path_dict = common_path(experiment)
    show_plot = False
    figure_dict = dict()
    step_index = -1
    figure_index = 0
    number_of_neurons = 10
    default_connection_strength = 1e-4
    KCs = AsymmetricalNodalCircuitExtraGround(number_of_neurons, default_connection_strength)

    # Test 1
    Test = 1
    connection_pairs = [[1, 0, 0.001],
                        [2, 5, 0.001],
                        [5, 3, 0.002],
                        [3, 5, 0.001],
                        [3, 1, 0.001],
                        [5, 7, 0.002],
                        [7, 0, 0.001]]  # 1 ohm from node 1 to 2
    KCs.set_connections(connection_pairs)
    potential_conservations = [[3, -1, 0.1]]  # ,[1, 0, 0] # 1 volt from node 2 to 0, in the direction of volt down
    KCs.add_potential_conservations(potential_conservations)
    KCs.set_inject_currents([[2, 1e-4]])
    KCs.set_target([[0, 1e-4]])
    for i in range(10):
        step_index += 1
        KCs.update_potential_diff()
        KCs.update_nodal_matrix()
        solves = KCs.solve()
        print("solves", solves)
    KCs.update_current()
    KCs.generate_graph()
    figure_dict[figure_index] = KCs.plot_graph(title="Test " + str(Test) + " Step " + str(step_index),
                                               show=show_plot)
    figure_index += 1
    KCs.update_potential_conservations([[3, -1, None]])
    for i in range(10):
        step_index += 1
        KCs.update_potential_diff()
        KCs.update_nodal_matrix()
        solves = KCs.solve()
        print("solves", solves)
    KCs.update_current()
    KCs.generate_graph()
    figure_dict[figure_index] = KCs.plot_graph(title="Test " + str(Test) + " Step " + str(step_index),
                                               show=show_plot)
    figure_index += 1

    # Test 2
    Test += 1
    step_index = -1
    for i in range(10):
        step_index += 1
        KCs.update_inject_currents([[5, 1e-5], [2, -1e-5]])
        KCs.update_potential_diff()
        KCs.update_nodal_matrix()
        solves = KCs.solve()
        print("solves", solves)
        KCs.update_current()
        KCs.generate_graph()
        figure_dict[figure_index] = KCs.plot_graph(title="Test " + str(Test) + " Step " + str(step_index),
                                                   show=show_plot)
        figure_index += 1
    for i in range(10):
        step_index += 1
        KCs.update_inject_currents([[7, 1e-5], [5, -1e-5]])
        KCs.update_potential_diff()
        KCs.update_nodal_matrix()
        solves = KCs.solve()
        print("solves", solves)
        KCs.update_current()
        KCs.generate_graph()
        figure_dict[figure_index] = KCs.plot_graph(title="Test " + str(Test) + " Step " + str(step_index),
                                                   show=show_plot)
        figure_index += 1
    for i in range(10):
        step_index += 1
        KCs.update_inject_currents([[0, 1e-5], [7, -1e-5]])
        KCs.update_potential_diff()
        KCs.update_nodal_matrix()
        solves = KCs.solve()
        print("solves", solves)
        KCs.update_current()
        KCs.generate_graph()
        figure_dict[figure_index] = KCs.plot_graph(title="Test " + str(Test) + " Step " + str(step_index),
                                                   show=show_plot)
        figure_index += 1
    # plt.show(block=False)
    save_plots_as_pdf(figure_dict, path_dict['plot_path'], file_name='plots.pdf')
    pass
