from Config.nodeWeight import NodeType
import networkx as nx
from Utils.matrix.matrixUtils import build_adjacent_matrix_from_weight_matrix, build_network_by_adjacent_matrix, \
    plot_network
import numpy as np
import matplotlib.pyplot as plt


# desc Map的基类
class RudiMapBasic(object):
    def __init__(self, weight_matrix, row, col):
        self._row = row
        self._col = col
        self._weight_m = weight_matrix
        self._adjacent_m = build_adjacent_matrix_from_weight_matrix(weight_matrix, row, col)
        self.network = build_network_by_adjacent_matrix(self._adjacent_m)
        self.navi_m = np.zeros((row, col))

    @property
    def weight_m(self):
        return self._weight_m

    @property
    def adjacent_m(self):
        return self._adjacent_m

    @property
    def row(self):
        return self._row

    @property
    def col(self):
        return self._col

    def add_start_point(self, x, y):
        self.navi_m[x, y] = NodeType.start_point
        self.network.node[x * self.col + y]["type"] = NodeType.start_point

    def add_end_point(self, x, y):
        self.navi_m[x, y] = NodeType.end_point
        self.network.node[x * self.col + y]["type"] = NodeType.end_point

    def add_pass_point(self, x, y):
        self.navi_m[x, y] = NodeType.pass_point
        self.network.node[x * self.col + y]["type"] = NodeType.pass_point

    def show_weight_m(self):
        plt.imshow(self.weight_m)
        plt.show()

    def show_navi_m(self):
        plt.imshow(self.navi_m)
        plt.show()

    def show_network(self):
        plot_network(self.network)
        plt.show()


if __name__ == '__main__':
    row = 5
    col = 6

    m = np.ones((row, col))
    m[0:3, 2] = 9
    m[2:4, 5] = 9

    rudi_map = RudiMapBasic(m, row, col)
    rudi_map.add_start_point(0, 0)
    rudi_map.add_end_point(4, 5)
    print(rudi_map.weight_m)
    print(rudi_map.adjacent_m)
    rudi_map.show_weight_m()
    rudi_map.show_navi_m()
    rudi_map.show_network()

    G = rudi_map.network
    print(nx.astar_path(G, 1, 15))
    print(nx.astar_path(G, 1, 15, weight='aa'))
    print(nx.bidirectional_dijkstra(G, 1, 15, weight='weight'))
    print(rudi_map.weight_m)
