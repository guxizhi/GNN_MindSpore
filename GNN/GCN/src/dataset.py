import numpy as np
import scipy.sparse as sp
import mindspore.dataset as ds


def normalize_adj(adj):
    """
    对称归一化邻接矩阵，以便后续计算
    """
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def get_adj_features_labels_mask(data_dir, train_nodes_num, eval_nodes_num, test_nodes_num):
    """
    获取邻接矩阵和节点特征、标签
    """
    # 读入以处理好的图数据
    g = ds.GraphData(data_dir)
    # 获取该图的所有结点
    nodes = g.get_all_nodes(0)
    nodes_list = nodes.tolist()
    # 对该图的所有结点进行邻居结点的采样，并获取节点特征
    row_tensor = g.get_node_feature(nodes_list, [1, 2])
    features = row_tensor[0]
    labels = row_tensor[1]

    nodes_num = labels.shape[0]

    train_mask = get_mask(nodes_num, 0, train_nodes_num)
    eval_mask = get_mask(nodes_num, train_nodes_num, train_nodes_num + eval_nodes_num)
    test_mask = get_mask(nodes_num, nodes_num - test_nodes_num, nodes_num)

    class_num = labels.max() + 1
    # 将各结点标签改为onehot形式
    labels_onehot = np.eye(nodes_num, class_num)[labels].astype(np.float32)

    # 获取邻接矩阵
    neighbor = g.get_all_neighbors(nodes_list, 0)
    node_map = {node_id: index for index, node_id in enumerate(nodes_list)}
    adj = np.zeros([nodes_num, nodes_num], dtype=np.float32)
    for index, value in np.ndenumerate(neighbor):
        if value >= 0 and index[1] > 0:
            adj[node_map[neighbor[index[0], 0]], node_map[value]] = 1
    adj = sp.coo_matrix(adj)
    adj = adj + adj.T.multiply(adj.T > adj) + sp.eye(nodes_num)
    # 对邻接矩阵进行对称归一化
    nor_adj = normalize_adj(adj)
    nor_adj = np.array(nor_adj.todense())
    return nor_adj, features, labels_onehot, labels, train_mask, test_mask, eval_mask

def load_and_process(data_dir, train_node_num, eval_node_num, test_node_num):
    """
    将各功能进行汇总，统一成函数load_and_process
    """
    nor_adj, features, labels_onehot, labels, train_mask, test_mask, eval_mask = get_adj_features_labels_mask(data_dir,
                                                                                                              train_node_num,
                                                                                                              eval_node_num,
                                                                                                              test_node_num)
    return nor_adj, features, labels_onehot, labels, train_mask, test_mask, eval_mask
    
def get_mask(total, begin, end):
    """
    生成mask
    """
    mask = np.zeros([total]).astype(np.float32)
    mask[begin:end] = 1
    return mask
