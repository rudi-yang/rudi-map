import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# desc Map的基类
from Utils.matrix.matrixUtils import get_adj_matrix_from_nx, plot_network


class RudiMapBasic(object):
    def __init__(self, row, col):
        self._row = row
        self._col = col
        self.G = nx.grid_2d_graph(row, col)

        self._pos = dict((n, n) for n in self.G.nodes())

        self._m_adj = get_adj_matrix_from_nx(self.G)

    @property
    def pos(self):
        return self._pos

    @property
    def adj(self):
        return self._m_adj

    @property
    def row(self):
        return self._row

    @property
    def col(self):
        return self._col

    def set_weight_random(self, matrix, region_m=None, ratio_m=None, attr="weight"):
        row = self.row
        col = self.col

        self._m_weight = np.triu(matrix) * [np.ones([row * col, row * col]) - np.eye(row * col, row * col)]
        self._m_weight = np.reshape(self._m_weight, [row * col, row * col])
        self._m_weight = self._m_weight + self._m_weight.T
        self._m_weight = np.multiply(self._m_weight, self._m_adj)

        if region_m is not None:
            self._m_weight = np.multiply(self._m_weight, region_m)

        if ratio_m is not None:
            self._m_weight = np.multiply(self._m_weight, ratio_m)

        m_weight = self._m_weight
        for r in range(0, m_weight.shape[0]):
            for c in range(r + 1, m_weight.shape[1]):
                if self.G.has_edge(list(self.G.nodes)[r], list(self.G.nodes)[c]):
                    self.G.edges[list(self.G.nodes)[r], list(self.G.nodes)[c]][attr] = m_weight[r, c]

    def show_network(self, attr="weight"):
        plot_network(self.G, self.pos, attr)


if __name__ == '__main__':
    row = 15
    col = 15

    rudi_map = RudiMapBasic(row, col)

    m_random = np.reshape(np.random.randint(1, 10, pow(row * col, 2)), [row * col, row * col])
    rudi_map.set_weight_random(m_random)

    rudi_map.show_network()
    path = np.array(nx.dijkstra_path(rudi_map.G, (0, 2), (14, 13)))
    plt.plot(path[:, 0], path[:, 1], lw=5, color="g")

    plt.savefig("z.png")
