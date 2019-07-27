import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


# task 对于整体连接的矩阵 n行m列
def build_adjacent_matrix_from_weight_matrix(matrix_w, row=3, col=6):
    if matrix_w.shape != (row, col):
        raise Exception("weight matrix shape not match")

    def get_adjacent_weight(i, j, col, matrix_w):
        row_i, col_i = int(i / col), i % col
        row_j, col_j = int(j / col), j % col

        diff_row = abs(row_i - row_j)
        diff_col = abs(col_i - col_j)

        if diff_row + diff_col == 1:
            print(row_i,row_j)
            print(col_i,col_j)
            return max(matrix_w[row_i, col_i], matrix_w[row_j, col_j])
        else:
            return 0

    adjacent_m = np.zeros((row * col, row * col))
    for i in range(0, row * col):
        for j in range(0, row * col):
            if get_adjacent_weight(i, j, col, matrix_w) > 0:
                adjacent_m[i, j] = get_adjacent_weight(i, j, col, matrix_w)

    return adjacent_m


# task 根据邻接矩阵生成network
def build_network_by_adjacent_matrix(adj_m):
    G = nx.Graph()
    (row, col) = adj_m.shape
    for r in range(0, row):
        for c in range(0, col):
            if adj_m[r, c] > 0:
                G.add_edge(r, c, weight=int(adj_m[r, c]))
    return G


# task 根据邻接矩阵作图
def plot_network(G):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    labes = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labes, font_color='red')
    plt.show()
