import numpy as np
import mindspore.dataset as ds


def get_features_labels_mask(data_dir, train_nodes_num, eval_nodes_num, test_nodes_num):
    """
    得到各节点的特征、标签，训练集、验证集、测试集
    """
    # 读入以处理好的图数据
    g = ds.GraphData(data_dir)
    # 获取该图的所有结点
    nodes = g.get_all_nodes(0)
    # 对该图的所有结点进行邻居结点的采样，这里采用多层采样
    nodes_and_neighbors = g.get_sampled_neighbors(nodes.tolist(), [10, 10], [0, 0]).tolist()
    # 获得所有邻居结点的特征
    row_tensor = g.get_node_feature(nodes_and_neighbors, [1, 2])

    features = row_tensor[0]
    labels = row_tensor[1]

    nodes_num = labels.shape[0]

    train_mask = get_mask(nodes_num, 0, train_nodes_num)
    eval_mask = get_mask(nodes_num, train_nodes_num, train_nodes_num + eval_nodes_num)
    test_mask = get_mask(nodes_num, nodes_num - test_nodes_num, nodes_num)

    class_num = labels.max() + 1

    return features, labels, train_mask, test_mask, eval_mask


def load_and_process(data_dir, train_node_num, eval_node_num, test_node_num):
    """
    将不同的数据获取函数，统一为load_and_process函数
    """
    features, labels, train_mask, test_mask, eval_mask = get_features_labels_mask(data_dir,
                                                                                  train_node_num,
                                                                                  eval_node_num,
                                                                                  test_node_num)
    return features, labels, train_mask, test_mask, eval_mask


def get_mask(total, begin, end):
    """
    生成mask
    """
    mask = np.zeros([total]).astype(np.float32)
    mask[begin:end] = 1
    return mask
