import sys
from mindspore import Parameter
from mindspore import Tensor
from mindspore import nn
from mindspore.ops import operations as P

sys.path.append("../..")
from layer.public.utils import get_kwargs


# FeaturePassing作为一个父类，所有GNNConv都继承于FeaturePassing
class FeaturePassing(nn.Cell):
    """
    完成对目标节点的邻居节点特征的聚合和传递
    """
    def __init__(self):
        super(FeaturePassing, self).__init__()
        # 直接定义聚合的方法
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.reduce_mean = P.ReduceMean()
        self.reduce_max = P.ReduceMax()
        self.reduce_min = P.ReduceMin()
        self.reduce_concat = P.Concat(-1)
        self.matmul = P.MatMul()

    def Propagate(self, input: Tensor, **kwargs) -> Tensor:
        """
        Propagte作为一个传播函数，直接完成邻居节点传递、聚合的工作
        将聚合后的返回值传递给目标节点进行关联,
        以下三个等式组合成函数Propagate功能
        """
        message_kwargs, aggregate_kwargs, update_kwargs = get_kwargs(**kwargs)
        neigbor_features = self.Message(input, **message_kwargs)
        neighbor_aggregate = self.Aggregate(neigbor_features,**aggregate_kwargs)
        neighbor_feature = self.Update(neighbor_aggregate, **update_kwargs)
        return neighbor_feature

    def Message(self, input: Tensor) -> Tensor:
        """
        Message作为传递函数，将邻居结点特征进行传递，默认直接传递，可重写
        将返回值输入Aggregate聚合函数，进行下一步聚合操作
        """
        return input

    def Aggregate(self, neighbor_features: Tensor, weight: Parameter, aggr_method='sum'):
        """
        Aggregate作为聚合函数，将邻居结点特征进行聚合，默认求和操作，可重写
        将返回值输入Upate更新函数，进行下一步操纵
        """
        # 对特征向量进行求和操作
        if aggr_method == "sum" or aggr_method == "add":
            aggr_feature = self.reduce_sum(neighbor_features, 1)
        # 对特征向量进行求平均操作
        elif aggr_method == "mean":
            aggr_feature = self.reduce_mean(neighbor_features)
        # 对特征向量进行求最小值操作
        elif aggr_method == "min":
            aggr_feature = self.reduce_min(neighbor_features)
        # 对特征向量进行求最大值操作
        elif aggr_method == "max":
            aggr_feature = self.reduce_max(neighbor_features)

        # 对输出特征向量进行线性变化，设定输出向量的维度
        output_feature = self.matmul(aggr_feature, weight)

        # if bias:
        #     output_feature += bias
        return output_feature

    def Update(self, input: Tensor) -> Tensor:
        """
        该步操作暂无实际意义，对于某些GNN可能需要重写
        """
        return input
