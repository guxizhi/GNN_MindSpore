import numpy as np
import mindspore.dataset as ds


def adj_to_bias(adj):
    """
    把自身结点加入到邻接矩阵中并保证只有一个hop参与计算
    """
    num_graphs = adj.shape[0]
    adj_temp = np.empty(adj.shape)
    for i in range(num_graphs):
        adj_temp[i] = adj[i] + np.eye(adj.shape[1])
    return -1e9 * (1.0 - adj_temp)


def get_biases_features_labels(data_dir):
    """
    得到邻接矩阵的偏差和各节点的特征、标签
    """
    # 读入以处理好的图数据
    g = ds.GraphData(data_dir)
    # 获取该图的所有结点
    nodes = g.get_all_nodes(0)
    nodes_list = nodes.tolist()
    # 对改图的所有结点进行邻居结点的采样，获得邻居结点特征
    row_tensor = g.get_node_feature(nodes_list, [1, 2])
    features = row_tensor[0]
    # 对特征向量进行增维操作，适应GAT算法
    features = features[np.newaxis]

    labels = row_tensor[1]

    nodes_num = labels.shape[0]
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
    adj = adj[np.newaxis]

    # 获得邻接矩阵偏差
    biases = adj_to_bias(adj)

    return biases, features, labels_onehot


def get_mask(total, begin, end):
    """
    生成mask
    """
    mask = np.zeros([total]).astype(np.float32)
    mask[begin:end] = 1
    return np.array(mask, dtype=np.bool)


def load_and_process(data_dir, train_node_num, eval_node_num, test_node_num):
    """
    将各功能进行汇总，统一成函数load_and_process
    """
    biases, feature, label = get_biases_features_labels(data_dir)

    # 对训练集、验证集、测试集进行划分
    nodes_num = label.shape[0]
    train_mask = get_mask(nodes_num, 0, train_node_num)
    eval_mask = get_mask(nodes_num, train_node_num, train_node_num + eval_node_num)
    test_mask = get_mask(nodes_num, nodes_num - test_node_num, nodes_num)

    y_train = np.zeros(label.shape)
    y_val = np.zeros(label.shape)
    y_test = np.zeros(label.shape)

    y_train[train_mask, :] = label[train_mask, :]
    y_val[eval_mask, :] = label[eval_mask, :]
    y_test[test_mask, :] = label[test_mask, :]

    y_train = y_train[np.newaxis]
    y_val = y_val[np.newaxis]
    y_test = y_test[np.newaxis]
    train_mask = train_mask[np.newaxis]
    eval_mask = eval_mask[np.newaxis]
    test_mask = test_mask[np.newaxis]

    return feature, biases, y_train, train_mask, y_val, eval_mask, y_test, test_mask
