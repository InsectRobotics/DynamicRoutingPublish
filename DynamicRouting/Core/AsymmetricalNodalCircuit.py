import copy
import matplotlib.pyplot as plt
import numpy as np
from DynamicRouting.Utils.MathUtil import matrix_minor, vector_minor
from DynamicRouting.Utils.pathselection import common_path
from DynamicRouting.Utils.MatrixAndOrders import expand_matrix, expand_vector, bisect_left
from DynamicRouting.Core.DirectedNeuronGraph import DirectedNeuronGraph
from DynamicRouting.Utils.plot import save_plots_as_pdf
import numba as nb


@nb.jit(nopython=True)
def update_potential_diff_numba(potential_vector):
    number_of_neurons = len(potential_vector)
    passable_connection = np.zeros((number_of_neurons, number_of_neurons), dtype=nb.boolean)
    potential_diff_matrix = np.zeros((number_of_neurons, number_of_neurons))
    for index1 in range(number_of_neurons):
        for index2 in range(index1):
            potential_diff_matrix[index1, index2] = potential_vector[index1] - potential_vector[index2]
            potential_diff_matrix[index2, index1] = -potential_diff_matrix[index1, index2]
            if index1 == index2:
                raise
            if potential_diff_matrix[index1, index2] >= 0:
                passable_connection[index1, index2] = True
                passable_connection[index2, index1] = False
            else:
                passable_connection[index1, index2] = False
                passable_connection[index2, index1] = True
    return potential_diff_matrix, passable_connection

class AsymmetricalNodalCircuit(DirectedNeuronGraph):
    """

    """

    def __init__(self, number_of_neurons, default_connection_strength):
        self.number_of_potential_conservations = 0
        self.number_of_neurons = number_of_neurons
        self.default_connection_strength = default_connection_strength
        self.connection_matrix = np.full([number_of_neurons, number_of_neurons], default_connection_strength,
                                         dtype=np.float)
        np.fill_diagonal(self.connection_matrix, 0)
        self.nodal_admittance_matrix = np.full([number_of_neurons, number_of_neurons], default_connection_strength,
                                               dtype=np.float)
        np.fill_diagonal(self.nodal_admittance_matrix, 0)
        np.fill_diagonal(self.nodal_admittance_matrix, -self.nodal_admittance_matrix.sum(axis=0))
        self.nodal_matrix = copy.deepcopy(self.nodal_admittance_matrix)
        self.conservation_vector = np.zeros(number_of_neurons)
        self.potential_vector = np.zeros(self.number_of_neurons)
        self.potential_conservations = dict()
        self.ground = dict()
        self.connection_pairs = dict()
        self.current_matrix = np.zeros([number_of_neurons, number_of_neurons])
        self.index_connected = np.arange(self.number_of_neurons)
        self.passable_connection = np.triu(np.ones((self.number_of_neurons, self.number_of_neurons), dtype=bool))
        self.potential_diff_matrix = np.zeros((self.number_of_neurons, self.number_of_neurons))

        self.update_potential_diff()
        self.update_nodal_matrix()


    def update_potential_diff(self):
        potential_diff_matrix = np.expand_dims(range(10), 1) - np.expand_dims(range(10), 0)
        passable_connection = np.greater_equal(self.potential_diff_matrix, 0)
        self.potential_diff_matrix, self.passable_connection = \
            update_potential_diff_numba(self.potential_vector)

    def update_current(self):
        self.current_matrix = self.potential_diff_matrix * self.connection_matrix
        self.current_matrix[np.logical_not(self.passable_connection)] = 0

    def update_nodal_matrix(self):
        self.nodal_admittance_matrix[self.passable_connection] \
            = -self.connection_matrix[self.passable_connection]
        self.nodal_admittance_matrix[self.passable_connection.T] \
            = self.nodal_admittance_matrix.T[self.passable_connection.T]
        np.fill_diagonal(self.nodal_admittance_matrix, 0)
        np.fill_diagonal(self.nodal_admittance_matrix, -self.nodal_admittance_matrix.sum(axis=0))
        self.nodal_matrix[:self.number_of_neurons, :self.number_of_neurons] = self.nodal_admittance_matrix

    def _update_current_conservation(self, i, j, conservation):
        self.conservation_vector[i] = conservation
        self.conservation_vector[j] = -conservation

    def _update_inject_current(self, i, current):
        self.conservation_vector[i] += current

    def _set_inject_current(self, i, current):
        self.conservation_vector[i] = current

    def _update_potential_conservation(self, k, conservation):
        self.conservation_vector[k] += conservation

    def _add_potential_conservation(self, i, j, k, conservation):
        self.conservation_vector[k] = conservation
        self.nodal_matrix[i, k] = 1
        self.nodal_matrix[j, k] = -1
        self.nodal_matrix[k, i] = 1
        self.nodal_matrix[k, j] = -1

    def update_current_conservations(self, conservation_changes):
        for i, j, conservation_change in conservation_changes:
            self._update_current_conservation(i, j, conservation_change)

    def update_inject_currents(self, currents):
        for i, current in currents:
            self._update_inject_current(i, current)

    def update_inject_currents_by_vector(self, currents_change_vector):
        self.conservation_vector[:self.number_of_neurons] += currents_change_vector

    def set_inject_currents_by_vector(self, currents_change_vector):
        self.conservation_vector[:self.number_of_neurons] = currents_change_vector

    def get_inject_current_vector(self):
        return self.conservation_vector[:self.number_of_neurons]

    def set_inject_currents(self, currents):
        for i, current in currents:
            self._set_inject_current(i, current)

    def add_potential_conservations(self, conservations, update_existing_state=True):
        self.conservation_vector = expand_vector(self.conservation_vector, len(conservations))
        self.nodal_matrix = expand_matrix(self.nodal_matrix, len(conservations))
        for i, j, conservation in conservations:
            if update_existing_state:
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

    def update_potential_conservations(self, conservation_changes):
        index_to_del = []
        for i, j, change in conservation_changes:
            if i in self.potential_conservations:
                if j in self.potential_conservations[i]:
                    if change is None:
                        index_to_del.append(self.potential_conservations[i][j]["index"])
                        self.potential_conservations[i].pop(j)
                        if not self.potential_conservations[i]:
                            self.potential_conservations.pop(i)
                    else:
                        self.potential_conservations[i][j]["value"] += change
                        self._update_potential_conservation(self.potential_conservations[i][j]["index"], change)
                else:
                    self.add_potential_conservations([[i, j, change]])
            else:
                self.add_potential_conservations([[i, j, change]])
        if index_to_del:
            index_to_del = np.sort(index_to_del)
            for k in index_to_del:
                self.conservation_vector = np.delete(self.conservation_vector, k)
                self.nodal_matrix = np.delete(self.nodal_matrix, k, axis=0)
                self.nodal_matrix = np.delete(self.nodal_matrix, k, axis=1)
            for i in self.potential_conservations.keys():
                for j in self.potential_conservations[i].keys():
                    self.potential_conservations[i][j]["index"] -= \
                        bisect_left(index_to_del, self.potential_conservations[i][j]["index"])
            self.number_of_potential_conservations -= len(index_to_del)

    def set_ground(self, ground=0):
        self.ground = ground
        self.nodal_minor = matrix_minor(self.nodal_matrix, ground, ground)
        self.conservation_minor = vector_minor(self.conservation_vector, ground)

    def solve(self):
        solves_minor = np.linalg.solve(self.nodal_minor, self.conservation_minor)
        self.solves = np.insert(solves_minor, self.ground, 0)
        self.potential_vector = self.solves[:self.number_of_neurons]
        return self.solves

    def plot_graph(self, title=None, block=False, show=True):
        fig = super().plot_graph(title=title, block=block, show=show,
                                 edge_colors_matrix=self.current_matrix,
                                 node_color_vector=self.potential_vector)
        return fig


if __name__ == "__main__":
    # Parameters
    experiment = 'AsymmetricalNodalCircuit'
    path_dict = common_path(experiment)
    figure_dict = dict()
    step_index = -1
    figure_index = 0
    number_of_neurons = 10
    default_connection_strength = 1e-4
    KCs = AsymmetricalNodalCircuit(number_of_neurons, default_connection_strength)

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
    potential_conservations = [[2, 0, 0.1],
                               [3, 0, 0.1]]  # ,[1, 0, 0] # 1 volt from node 2 to 0, in the direction of volt down
    KCs.add_potential_conservations(potential_conservations)
    for i in range(10):
        step_index += 1
        KCs.update_potential_diff()
        KCs.update_nodal_matrix()
        KCs.set_ground(0)
        solves = KCs.solve()
        print("solves", solves)
    KCs.update_current()
    KCs.generate_graph()
    figure_dict[figure_index] = KCs.plot_graph(title="Test " + str(Test) + " Step " + str(step_index))
    figure_index += 1
    KCs.update_potential_conservations([[3, 0, None], [9, 0, 0.1]])
    for i in range(10):
        step_index += 1
        KCs.update_potential_diff()
        KCs.update_nodal_matrix()
        KCs.set_ground(0)
        solves = KCs.solve()
        print("solves", solves)
    KCs.update_current()
    KCs.generate_graph()
    figure_dict[figure_index] = KCs.plot_graph(title="Test " + str(Test) + " Step " + str(step_index))
    figure_index += 1

    # Test 2
    Test += 1
    step_index = -1
    KCs.update_potential_conservations([[9, 0, None]])
    for i in range(10):
        step_index += 1
        KCs.update_potential_conservations([[5, 0, 0.01], [2, 0, -0.01]])
        KCs.update_potential_diff()
        KCs.update_nodal_matrix()
        KCs.set_ground(0)
        solves = KCs.solve()
        print("solves", solves)
        KCs.update_current()
        KCs.generate_graph()
        figure_dict[figure_index] = KCs.plot_graph(title="Test " + str(Test) + " Step " + str(step_index))
        figure_index += 1
    for i in range(10):
        step_index += 1
        KCs.update_potential_conservations([[7, 0, 0.01], [5, 0, -0.01]])
        KCs.update_potential_diff()
        KCs.update_nodal_matrix()
        KCs.set_ground(0)
        solves = KCs.solve()
        print("solves", solves)
        KCs.update_current()
        KCs.generate_graph()
        figure_dict[figure_index] = KCs.plot_graph(title="Test " + str(Test) + " Step " + str(step_index))
        figure_index += 1
    current = 0
    for i in range(10):
        step_index += 1
        current += 2e-5
        KCs.update_inject_currents([[3, current]])
        KCs.update_potential_diff()
        KCs.update_nodal_matrix()
        KCs.set_ground(0)
        solves = KCs.solve()
        print("solves", solves)
        KCs.update_current()
        KCs.generate_graph()
        figure_dict[figure_index] = KCs.plot_graph(title="Test " + str(Test) + " Step " + str(step_index))
        figure_index += 1
    plt.show(block=False)
    save_plots_as_pdf(figure_dict, path_dict['plot_path'], file_name='plots.pdf')
    pass
