import sys
import mindspore
from mindspore import nn
from mindspore import Parameter
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.initializer import initializer
from mindspore.nn.layer.activation import get_activation

# sys.path.append("..")
sys.path.append("../..")
from layer.public.layerNeed import MLP, AttentionHead
from layer.public.feature_pass import FeaturePassing
from layer.public.utils import glorot


class GATConv(FeaturePassing):
    """
        对GAT的传播层进行定义
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 heads=1,
                 in_drop=0.0,
                 coef_drop=0.0,
                 activation=nn.ELU(),
                 residual=False,
                 aggr_method="concat",
                 update_method="direct"):
        super(GATConv, self).__init__()
        assert aggr_method in ["mean", "sum", "concat"]
        assert update_method in ["concat", "sum", "direct"]
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.heads = heads
        self.weight = Parameter(default_input=Tensor(glorot(self.input_dim, self.output_dim)), name='w')
        self.attns = []

        # 添加多头注意力机制，以增加可信度
        for _ in range(heads):
            self.attns.append(AttentionHead(input_dim,
                                            output_dim,
                                            in_drop_ratio=in_drop,
                                            coef_drop_ratio=coef_drop,
                                            activation=activation,
                                            residual=residual))
        self.attns = nn.layer.CellList(self.attns)
        if aggr_method == 'concat':
            self.out_trans = P.Concat(-1)
        elif aggr_method == 'sum':
            self.out_trans = P.AddN()
        self.update_method = update_method

    def construct(self, node_features: Tensor, biases: Tensor, training):
        # Propagate融合了Message、Aggregate、Update的功能
        neighbor_feature = self.Propagate(input=node_features, weight=self.weight, aggr_method="concat",
                                          bias_mat=biases, training=training)

        # GAT直接在Message中进行与自身结点的关联，或者直接使用聚合特征
        if self.update_method == "direct":
            hidden = neighbor_feature
        # print(hidden.shape)
        return hidden

    def Message(self, input: Tensor, bias_mat: Tensor, training):
        # GAT需要对Message函数进行重写，完成多头注意力机制
        # 并将多头的邻居节点特征传递给Aggregate函数，进行聚合操作
        # print("rewrite message")
        res = ()
        for i in range(self.heads):
            res += (self.attns[i](input, bias_mat, training),)
        return res

    def Aggregate(self, neighbor_features: Tensor, weight: Parameter, aggr_method: str):
        # GAT需要对Aggregate函数进行重写，注意这里的聚合是对多头的聚合
        # print("rewrite aggregate")
        if aggr_method == "concat":
            aggr_feature = self.reduce_concat(neighbor_features)
        output_feature = aggr_feature
        # print(output_feature.shape)
        return output_feature


class GCNConv(FeaturePassing):
    """
    对GCN的传播层进行定义
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 activation=None,
                 dropout_ratio=None,
                 aggr_method="sum",
                 update_method="direct"):
        super(GCNConv, self).__init__()

        # 对于邻居聚合方式和与目标节点关联方式进行限制
        assert aggr_method in ["mean", "sum", "concat"]
        assert update_method in ["concat", "sum", "direct"]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.aggr_neighbor_method = aggr_method
        self.aggr_hidden_method = update_method
        self.activation = activation

        # 对于可训练矩阵进行初始化、定义
        self.weight = Tensor(glorot(self.output_dim, self.input_dim))
        self.fuction = nn.Dense(self.input_dim, self.output_dim, weight_init=self.weight, has_bias=False)

        # 对训练细节进行定义
        self.dropout_ratio = dropout_ratio
        if self.dropout_ratio is not None:
            self.dropout = nn.Dropout(keep_prob=1-self.dropout_ratio)
        self.dropout_flag = self.dropout_ratio is not None
        self.training = self.dropout_ratio is not None
        self.activation = get_activation(activation)
        self.activation_flag = self.activation is not None
        self.matmul = P.MatMul()


    def construct(self, adj, node_features):
        # Propagate融合了Message、Aggregate、Update的功能
        neighbor_feature = self.Propagate(input=node_features, adj=adj, weight=self.weight, aggr_method="sum")

        # GCN直接在Aggregate中进行与自身结点的关联，或者直接使用聚合特征
        if self.aggr_hidden_method == "direct":
            hidden = neighbor_feature
        # print(hidden.shape)
        return hidden

    def Aggregate(self, node_features, adj, weight, aggr_method):
        # GAT需要对Aggregate函数进行重写，注意这里的聚合根据邻接矩阵进行运算
        # print("rewrite aggregate!")
        if self.dropout_flag:
            node_features = self.dropout(node_features)

        fc = self.fuction(node_features)
        output_feature = self.matmul(adj, fc)

        if self.activation_flag:
            output_feature = self.activation(output_feature)
        return output_feature


class GINConv(FeaturePassing):
    """
    对GIN的传播层进行定义，目前的GIN是一个实用与节点分类的版本
    后续更新数据集后，补充适用于图分类版本，添加readout函数对节点特征进行加和
    """
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 eps: float = 0.,
                 train_eps=False,
                 hops=[10, 10],
                 activation="Relu",
                 aggr_method="sum",
                 update_method="concat",
                 has_bias=False):
        super(GINConv, self).__init__()

        # 对于邻居聚合方式和与目标节点关联方式进行限制
        assert aggr_method in ["mean", "sum"]
        assert update_method in ["concat", "sum"]

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hops = hops
        self.has_bias = has_bias
        self.aggr_neighbor_method = aggr_method
        self.aggr_hidden_method = update_method

        # 判断是否对eps进行训练
        if train_eps:
            self.eps = Parameter(eps)
        else:
            self.eps = eps

        self.activation = activation
        if activation == "Relu":
            self.activation = nn.ReLU()

        # 对于可训练矩阵、多层感知机进行初始化、定义
        self.mlp = MLP(input_dim=self.output_dim, hidden_dim=self.hidden_dim, output_dim=self.output_dim)
        self.weight = Parameter(default_input=Tensor(glorot(self.input_dim, self.hidden_dim)), name='w')
        self.weight1 = Parameter(default_input=Tensor(glorot(self.input_dim, self.hidden_dim)), name='w1')
        self.weight2 = Parameter(default_input=Tensor(glorot(self.hidden_dim, self.output_dim)), name='w2')
        self.weight3 = Parameter(default_input=Tensor(glorot(self.hidden_dim, self.hidden_dim)), name='w3')
        self.weight4 = Parameter(default_input=Tensor(glorot(self.hidden_dim, self.output_dim)), name='w4')
        if has_bias:
            self.bias = Parameter(Tensor(self.out_dim), name='b')
        self.matmul = P.MatMul()
        self.update_method = P.Concat(-1)

    def construct(self, node_features):
        # 对目标结点进行线性变化，使其与聚合后向量维度保持一致
        self_feature = self.matmul(Tensor(node_features[:, 0].asnumpy(), mindspore.float32), self.weight1)

        # 计算每一个hop所要聚合的结点个数，并添加到features_to_aggr中
        features_to_aggr = []
        hops = [1] + self.hops
        sum = 1
        features_to_aggr.append(sum)

        # 对每一个hop的邻居结点进行聚合
        for i in range(len(hops) - 1):
            sum += hops[i] * hops[i + 1]
            features_to_aggr.append(sum)
        for i in range(len(self.hops), 1, -1):
            # Propagate融合了Message、Aggregate、Update的功能
            neighbor_feature = self.Propagate(input=node_features[:, features_to_aggr[i - 1]:features_to_aggr[i]],
                                              weight=self.weight, aggr_method="sum")

        # 这里体现出GIN的不同，乘以参数（1+eps）
        hidden = (1 + self.eps) * self_feature + neighbor_feature
        hidden = self.matmul(hidden, self.weight2)
        if self.has_bias:
            hidden += self.bias
        # hidden = self.mlp(hidden)，mlp算子原因暂没有测试，用两个隐藏层代替
        hidden = self.matmul(hidden, self.weight3)
        if self.activation:
            hidden = self.activation(hidden)
        hidden = self.matmul(hidden, self.weight4)

        return hidden


class SageConv(FeaturePassing):
    """
    对Graphsage的传播层进行定义
    """
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hops=[10, 10],
                 activation="Relu",
                 aggr_method="sum",
                 update_method="concat",
                 has_bias=False):
        super(SageConv, self).__init__()

        # 对于邻居聚合方式和与目标节点关联方式进行限制
        assert aggr_method in ["mean", "sum"]
        assert update_method in ["concat", "sum"]

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hops = hops
        self.aggr_neighbor_method = aggr_method
        self.aggr_hidden_method = update_method
        self.activation = activation
        if activation == "Relu":
            self.activation = nn.ReLU()

        # 对于可训练矩阵进行初始化、定义
        self.weight = Parameter(default_input=Tensor(glorot(self.input_dim, self.hidden_dim)), name='w')
        self.weight1 = Parameter(default_input=Tensor(glorot(self.input_dim, self.hidden_dim)), name='w1')
        self.weight2 = Parameter(default_input=Tensor(glorot(self.hidden_dim, self.output_dim)), name='w2')
        self.weight3 = Parameter(default_input=Tensor(glorot(self.hidden_dim * 2, self.output_dim)), name='w3')
        # if self.has_bias:
        #     self.bias = Parameter(Tensor(self.out_dim), name='b')

        self.matmul = P.MatMul()
        self.update_method = P.Concat(-1)

    def construct(self, node_features):
        # 对目标结点进行线性变化，使其与聚合后向量维度保持一致
        self_feature = self.matmul(Tensor(node_features[:, 0].asnumpy(), mindspore.float32), self.weight1)

        # 计算每一个hop所要聚合的结点个数，并添加到features_to_aggr中
        features_to_aggr = []
        hops = [1] + self.hops
        sum = 1
        features_to_aggr.append(sum)
        for i in range(len(hops)-1):
            sum += hops[i] * hops[i+1]
            features_to_aggr.append(sum)

        # 对每一个hop的邻居结点进行聚合
        for i in range(len(self.hops), 1, -1):
            # Propagate融合了Message、Aggregate、Update的功能
            # Graphsage在聚合函数上使用求和的方法
            neighbor_feature = self.Propagate(input=node_features[:, features_to_aggr[i-1]:features_to_aggr[i]],
                                              weight=self.weight, aggr_method="sum")
            # 对于graphsage的两种与目标结点进行联系的方法
            if self.aggr_hidden_method == "sum":
                hidden = self_feature + neighbor_feature
                hidden = self.matmul(hidden, self.weight2)
            elif self.aggr_hidden_method == "concat":
                # concat之后维度会发生变化 * 2，使用weight3使其保持维度一致
                hidden = self.update_method((self_feature, neighbor_feature))
                hidden = self.matmul(hidden, self.weight3)

        # 输入激活函数
        if self.activation:
            return self.activation(hidden)
        else:
            return hidden