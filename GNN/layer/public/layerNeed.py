import sys
from mindspore import nn
from mindspore import Parameter
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.initializer import initializer


class GNNFeatureTransform(nn.Cell):
    """
    对特征进行线性变化，以便后续计算，用于GAT的实现
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True):
        super(GNNFeatureTransform, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.has_bias = has_bias

        # 判断权重矩阵是否符合初始化要求
        if isinstance(weight_init, Tensor):
            if weight_init.ndim != 2 or weight_init.shape[0] != output_dim or weight_init.shape[1] != input_dim:
                raise ValueError("weight_init shape error")

        self.weight = Parameter(initializer(weight_init, [output_dim, input_dim]))

        # 判断偏置矩阵是否符合初始化要求
        if self.has_bias:
            if isinstance(bias_init, Tensor):
                if bias_init.ndim != 1 or bias_init.shape[0] != output_dim:
                    raise ValueError("bias_init shape error")

            self.bias = Parameter(initializer(bias_init, [output_dim]))

        self.matmul = P.MatMul(transpose_b=True)
        self.bias_add = P.BiasAdd()

    def construct(self, x):
        tensor_shape = F.shape(x)
        input_feature = F.reshape(x, (tensor_shape[0] * tensor_shape[1], tensor_shape[2]))
        output = self.matmul(input_feature, self.weight)
        if self.has_bias:
            output = self.bias_add(output, self.bias)
        output = F.reshape(output, (tensor_shape[0], tensor_shape[1], self.output_dim))
        return output


class AttentionHead(nn.Cell):
    """
    使用AttentionHead类完成多头注意力机制，用于GAT的实现
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 in_drop_ratio=0.0,
                 coef_drop_ratio=0.0,
                 residual=False,
                 coef_activation=nn.LeakyReLU(),
                 activation=nn.ELU()):
        super(AttentionHead, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.in_drop_ratio = in_drop_ratio
        self.in_drop = nn.Dropout(keep_prob=1 - in_drop_ratio)
        self.in_drop_2 = nn.Dropout(keep_prob=1 - in_drop_ratio)
        self.feature_transform = GNNFeatureTransform(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            has_bias=False,
            weight_init='XavierUniform')

        self.f_1_transform = GNNFeatureTransform(
            input_dim=self.output_dim,
            output_dim=1,
            weight_init='XavierUniform')
        self.f_2_transform = GNNFeatureTransform(
            input_dim=self.output_dim,
            output_dim=1,
            weight_init='XavierUniform')
        self.softmax = nn.Softmax()

        self.coef_drop = nn.Dropout(keep_prob=1 - coef_drop_ratio)
        self.matmul = P.MatMul()
        self.bias_add = P.BiasAdd()
        self.bias = Parameter(initializer('zeros', self.output_dim))
        self.residual = residual
        if self.residual:
            if input_dim != output_dim:
                self.residual_transform_flag = True
                self.residual_transform = GNNFeatureTransform(
                    input_dim=self.input_dim,
                    output_dim=self.output_dim)
            else:
                self.residual_transform = None
        self.coef_activation = coef_activation
        self.activation = activation

    def construct(self, input, biases, training=True):
        if training is True:
            input = self.in_drop(input)

        feature = self.feature_transform(input)
        # 注意力机制的计算
        f_1 = self.f_1_transform(feature)
        f_2 = self.f_2_transform(feature)
        logits = f_1 + P.Transpose()(f_2, (0, 2, 1))
        logits = self.coef_activation(logits) + biases
        coefs = self.softmax(logits)
        if training is True:
            coefs = self.coef_drop(coefs)
            feature = self.in_drop_2(feature)

        coefs = P.Squeeze(0)(coefs)
        feature = P.Squeeze(0)(feature)

        ret = self.matmul(coefs, feature)
        ret = self.bias_add(ret, self.bias)
        ret = P.ExpandDims()(ret, 0)

        # 是否加入自身特征
        if self.residual:
            if self.residual_transform_flag:
                res = self.residual_transform(input)
                ret = ret + res
            else:
                ret = ret + input

        # 输入激活函数
        if self.activation is not None:
            ret = self.activation(ret)
        return ret


class MLP(nn.Cell):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2) -> object:
        """
        类MLP实现多层感知机的功能，用于GIN的实现
        """
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.Linear00 = nn.Dense(input_dim, hidden_dim, weight_init='normal', has_bias=False)
        self.Linear01 = nn.Dense(hidden_dim, hidden_dim, weight_init='normal', has_bias=False)
        self.Linear02 = nn.Dense(hidden_dim, output_dim, weight_init='normal', has_bias=False)

        # 只在Ascend平台上支持,CPU和GPU均不支持，后期进行测试
        # self.BatchNorm00 = P.BatchNorm1d(num_features=output_dim, is_training=True)

        self.ReLU00 = nn.ReLU()
        self.ReLU01 = nn.ReLU()
        self.ReLU02 = nn.ReLU()

    def construct(self, x):
        input = x
        hidden1 = self.Linear00(input)
        # batchnorm1 = self.BatchNorm00(hidden1)
        # relu1 = self.ReLU00(batchnorm1)
        relu1 = self.ReLU00(hidden1)
        hidden2 = self.Linear01(relu1)
        # batchnorm2 = self.BatchNorm01(hidden2)
        # relu2 = self.ReLU01(batchnorm2)
        relu2 = self.ReLU01(hidden2)
        output = self.Linear02(relu2)

        return output