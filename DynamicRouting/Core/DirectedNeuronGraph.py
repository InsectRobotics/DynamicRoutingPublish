import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import networkx as nx
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from DynamicRouting.Utils.pathselection import common_path
from matplotlib.legend_handler import HandlerPatch
from DynamicRouting.Utils.plot import make_legend_arrow, kmeans_representer
import matplotlib.patches as mpatches


class DirectedNeuronGraph:
    """

    """

    def __init__(self, number_of_neurons, default_connection_strength):
        self.number_of_neurons = number_of_neurons
        self.default_connection_strength = default_connection_strength
        self.connection_matrix = np.full([number_of_neurons, number_of_neurons], default_connection_strength, dtype=float)
        np.fill_diagonal(self.connection_matrix, 0)
    #
    # def set_default_connection(self, default_connection_strength, index_from=None, index_to=None):
    #     self.connection_matrix[index_from, :] = default_connection_strength

    def _update_connection(self, i, j, connection):
        self.connection_matrix[i, j] += connection

    def _set_connection(self, i, j, connection):
        self.connection_matrix[i, j] = connection

    def update_connections(self, connection_strength_changes):
        for i, j, connection_strength_change in connection_strength_changes:
            self._update_connection(i, j, connection_strength_change)

    def update_connections_with_matrix(self, connection_strength_change_matrix):
        self.connection_matrix += connection_strength_change_matrix

    def connections_matrix_normalise(self):
        self.connection_matrix /= self.connection_matrix.sum()

    def set_connections(self, connection_strengthes):
        for i, j, connection_strength in connection_strengthes:
            if i is None:
                self.connection_matrix[:, j] = connection_strength
            elif j is None:
                self.connection_matrix[i, :] = connection_strength
            else:
                self._set_connection(i, j, connection_strength)

    def generate_graph(self, number_of_edges=100):
        if self.number_of_neurons >= 20:
            self.for_plot_graph = nx.MultiDiGraph()
            self.for_plot_graph.add_nodes_from(range(self.number_of_neurons))
            edge_index_from_list = []
            edge_index_to_list = []
            edge_weight_list = []
            for i1 in range(self.number_of_neurons):
                for i2 in range(self.number_of_neurons):
                    if self.connection_matrix[i1, i2] > self.default_connection_strength:
                        edge_index_from_list.append(i1)
                        edge_index_to_list.append(i2)
                        edge_weight_list.append(self.connection_matrix[i1, i2])
            if len(edge_weight_list) > number_of_edges:
                sorted_index = np.argsort(edge_weight_list)
                edge_index_from_list = list(np.array(edge_index_from_list)[sorted_index[-number_of_edges:]])
                edge_index_to_list = list(np.array(edge_index_to_list)[sorted_index[-number_of_edges:]])
                edge_weight_list = list(np.array(edge_weight_list)[sorted_index[-number_of_edges:]])
            # self.for_plot_edge_index_from_list = edge_index_from_list
            # self.for_plot_edge_index_to_list = edge_index_to_list
            # self.for_plot_edge_weight_list = edge_weight_list
            self.for_plot_graph.add_weighted_edges_from(zip(edge_index_from_list, edge_index_to_list, edge_weight_list))
            # self.graph.add_edge(i1, i2, weight=self.connection_matrix[i1, i2])
        else:
            self.for_plot_graph = nx.convert_matrix.from_numpy_matrix(self.connection_matrix, create_using=nx.MultiDiGraph)

    def plot_graph(self, title=None, block=False, show=True, edge_colors_matrix=None, node_color_vector=None, edge_label_texts=None,
                   edgewidth_method='proportional'):
        self.plot_graph_handles = {}
        def legend_Kmeans(orig_handle, labels, n_clusters=5):
            import matplotlib.patches as mpatches
            from matplotlib.legend_handler import HandlerPatch
            if len(labels) > 5:
                closet_index = kmeans_representer(np.expand_dims(labels, axis=1), n_clusters=n_clusters)
                labels2 = labels[closet_index]
                orig_handle2 = [orig_handle[i] for i in closet_index]
            else:
                labels2 = labels
                orig_handle2 = orig_handle
            labels2Str = ["{:.2e}".format(alabel) for alabel in labels2]
            plt.legend(orig_handle2, labels2Str,
                       loc='upper left',
                       bbox_to_anchor=(-0.1, 1.1),
                       handler_map={mpatches.FancyArrowPatch: HandlerPatch(patch_func=make_legend_arrow), })

        fig, ax = plt.subplots()
        # pos = nx.kamada_kawai_layout(self.graph)

        edge_labels = dict([((n1, n2), d['weight'])
                            for n1, n2, d in self.for_plot_graph.edges(data=True)])
        # edge width is proportional number of games played
        weights = np.array([self.for_plot_graph.get_edge_data(u, v)[0]['weight'] for u, v in self.for_plot_graph.edges()])

        if edgewidth_method == 'log':
            b = 1  # when wight is 1, the edgewidth is 1
            a = (0.1 - b) / np.log10(self.default_connection_strength)  # when wight is 1e-10, the edgewidth is 0.1
            edgewidth = (a * np.log10(weights) + b) * 10
        elif edgewidth_method == 'proportional':
            edgewidth = weights / weights.max() * 10
        if self.number_of_neurons > 20:
            pos = nx.spring_layout(self.for_plot_graph)
        else:
            pos = nx.circular_layout(self.for_plot_graph)
        # self.graph_figure = nx.draw(self.graph, labels=labels, connectionstyle='arc3, rad = 0.1')
        if node_color_vector is None:
            node_color_vector = 'k'
        nodes = nx.draw_networkx_nodes(self.for_plot_graph, pos, node_color=node_color_vector,
                                       node_size=300)  # node_color="blue"
        # nx.draw(self.graph, with_labels=True, connectionstyle='arc3, rad = 0.1')
        if edge_colors_matrix is None:
            edge_colors = 'k'
        else:
            edge_colors_matrix = np.array(edge_colors_matrix)
            assert edge_colors_matrix.shape == (self.number_of_neurons, self.number_of_neurons)
            edge_colors = [edge_colors_matrix[n1, n2] for n1, n2 in self.for_plot_graph.edges()]

        edge_cmap = mpl.cm.get_cmap('copper')
        edge_alpha = 1

        edges = nx.draw_networkx_edges(
            self.for_plot_graph,
            pos,
            arrowstyle="->",
            arrowsize=20,
            # edge_color=edge_color_deepest,
            alpha=edge_alpha,
            edge_color=edge_colors,
            edge_cmap=edge_cmap,
            width=edgewidth,
            connectionstyle='arc3, rad = 0.1'
        )

        legend_Kmeans(edges, weights)

        if isinstance(edge_colors_matrix, (np.ndarray)):
            # set alpha value for each edge
            edge_colors_max = max(edge_colors)
            if edge_colors_max == 0:
                edge_colors_max = 1
            edge_alphas = np.array(edge_colors) / edge_colors_max
            for index in range(len(edgewidth)):
                edges[index].set_alpha(0.1 if edge_alphas[index] <= 0.0001 else 1)
        labels = dict(zip(range(self.number_of_neurons), range(self.number_of_neurons)))
        nx.draw_networkx_labels(self.for_plot_graph, pos=pos, labels=labels, font_color='w')

        if edge_label_texts is not None:
            edge_labels = nx.draw_networkx_edge_labels(self.for_plot_graph, pos=pos,
                                                       edge_labels=edge_label_texts, ax=ax, font_size=7, alpha=0.8,
                                                       verticalalignment='center', label_pos=0.3)

        plt.axis("off")
        ax_node_colorbar = inset_axes(ax,
                                      width="5%",  # width = 5% of parent_bbox width
                                      height="40%",  # height : 50%
                                      loc='upper left',
                                      bbox_to_anchor=(0.95, 0., 1, 1),
                                      bbox_transform=ax.transAxes,
                                      borderpad=0,
                                      )

        node_colorbar = plt.colorbar(nodes, cax=ax_node_colorbar, shrink=1)
        node_colorbar.set_label('Potential')
        node_colorbar.draw_all()

        ax_edge_colorbar = inset_axes(ax,
                                      width="5%",  # width = 5% of parent_bbox width
                                      height="40%",  # height : 50%
                                      loc='lower left',
                                      bbox_to_anchor=(0.95, 0., 1, 1),
                                      bbox_transform=ax.transAxes,
                                      borderpad=0,
                                      )
        if isinstance(edge_colors_matrix, (np.ndarray)):
            pc = mpl.collections.PatchCollection(edges, cmap=edge_cmap)
            pc.set_array(edge_colors)
            edge_colorbar = plt.colorbar(pc, cax=ax_edge_colorbar, shrink=1)
            edge_colorbar.set_alpha(edge_alpha)
            edge_colorbar.draw_all()
            edge_colorbar.set_label('Current')

        # Resize figure for label readibility
        l, b, w, h = ax.get_position().bounds
        ax.set_position([l / 2, b, w, h])
        if title is not None:
            fig.suptitle(title)
        if show:
            plt.show(block=block)
        return fig # , ax, pos

    def save_graph(self, path='graph.csv'):
        nx.readwrite.edgelist.write_edgelist(self.for_plot_graph, path)


if __name__ == "__main__":
    # Parameters
    experiment = 'DirectGraph'
    path_dict = common_path(experiment)
    figure_dict = dict()
    step_index = -1
    figure_index = 0
    number_of_neurons = 21
    default_connection_strength = 1e-4
    KCs = DirectedNeuronGraph(number_of_neurons, default_connection_strength)

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
    KCs.generate_graph(number_of_edges=5)
    figure_dict[figure_index] = KCs.plot_graph(title="Test " + str(Test) + " Step " + str(step_index))
    plt.show(block=False)
    pass
