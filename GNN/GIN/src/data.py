import mindspore
from mindspore import Tensor
import numpy as np
from ast import literal_eval

class Molecule:
    def __init__(self, graph, nodes):
        self.graph = graph # Ajacency Matrix
        self.nodes = nodes # means C, N, ... Br

def load_data():
    # return: list of Molecule
    data_num = 1112
    graph_label_file = open('../dataset/PROTEINS/PROTEINS_graph_labels.txt')
    node_label_file = open('../dataset/PROTEINS/PROTEINS_node_labels.txt')
    node_attribute_file = open('../dataset/PROTEINS/PROTEINS_node_attributes.txt')
    graph_idx_file = open('../dataset/PROTEINS/PROTEINS_graph_indicator.txt')
    adjacency_file = open('../dataset/PROTEINS/PROTEINS_A.txt')

    # check graph size
    tmp0, tmp1 = 1, 1
    # list of adjacency matrix
    graph_list = []
    # list of total number of nodes that each graph has
    graph_size_list = []

    for graph_idx in range(data_num):
        # total number of nodes that graph has
        graph_size = 1

        while True:
            tmp1 = int(graph_idx_file.readline())
            if tmp0 == tmp1:
                graph_size += 1
                tmp0 = tmp1
            else:
                tmp0 = tmp1
                break
        # print(graph_size)
        graph_list.append(np.zeros([graph_size, graph_size], dtype=np.float32))
        graph_size_list.append(graph_size)

    # check adjacency matrix
    tmp_sum = 0
    tmp_sum1 = 0
    for graph_idx in range(data_num):
        tmp_sum += graph_size_list[graph_idx]

        while True:
            edge = literal_eval(adjacency_file.readline())
            if (edge[0] <= tmp_sum) & (edge[1] <= tmp_sum):
                graph_list[graph_idx][edge[0] - tmp_sum1][edge[1] - tmp_sum1] = 1.0
                graph_list[graph_idx][edge[1] - tmp_sum1][edge[0] - tmp_sum1] = 1.0
            else:
                break
        tmp_sum1 = tmp_sum

    graph_list_tensor = []
    for graph_idx in range(data_num):
        graph_list_tensor.append(Tensor(graph_list[graph_idx]))
    # print(graph_list_tensor)

    # check node feature
    node_dim = 4
    node_list = []
    for graph_idx in range(data_num):
        nodes = np.zeros([graph_size_list[graph_idx], node_dim], dtype=np.float32)
        for node_idx in range(graph_size_list[graph_idx]):
            node_lable = int(node_label_file.readline())
            node_attribute = float(node_attribute_file.readline())
            node_val = 1.0
            if node_lable == 0:
                nodes[node_idx] = [node_val, 0.0, 0.0, node_attribute]
            elif node_lable == 1:
                nodes[node_idx] = [0.0, node_val, 0.0, node_attribute]
            else:
                nodes[node_idx] = [0.0, 0.0, node_val, node_attribute]
        node_list.append(Tensor(nodes, mindspore.float32))

    # set molecule class
    molecule_list = []
    label_list = []
    for graph_idx in range(data_num):
        molecule_list.append(Molecule(graph_list_tensor[graph_idx], node_list[graph_idx]))
        label_list.append(int(graph_label_file.readline()) - 1)

    graph_label_file.close()
    node_label_file.close()
    node_attribute_file.close()
    graph_idx_file.close()
    adjacency_file.close()

    return molecule_list, label_list

data = load_data()
print(data)