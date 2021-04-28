import numpy as np


def glorot(row, col):
    """
    glorot函数用来对权重矩阵进行初始化的工作,
    输入矩阵的长和宽，返回一个np数组，使用时改为mindspore的Tensor类型
    """
    range = np.sqrt(6.0 / (row + col))
    init = np.random.uniform(-range, range, [row, col]).astype(np.float32)
    return init


def get_kwargs(**kwargs):
    """
    get_kwargs函数用来将propagate函数的参数进行分离，
    分别按类型传给message、aggregate、update，
    输入**kwargs类型，返回字典类型的参数，可以直接输入当作函数参数
    """
    message_kwargs = {}
    aggregate_kwargs = {}
    update_kwargs = {}
    for i in kwargs:
        # 对不同参数进行分类，如有需要可以扩充
        if i == 'aggr_method':
            aggregate_kwargs['aggr_method'] = kwargs['aggr_method']
        elif i == 'adj':
            aggregate_kwargs['adj'] = kwargs['adj']
        elif i == 'weight':
            aggregate_kwargs['weight'] = kwargs['weight']
        elif i == 'bias_mat':
            message_kwargs['bias_mat'] = kwargs['bias_mat']
        elif i == 'training':
            message_kwargs['training'] = kwargs['training']
    return message_kwargs, aggregate_kwargs, update_kwargs