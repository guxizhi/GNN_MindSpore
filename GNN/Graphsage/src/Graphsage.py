import mindspore
import numpy as np
from mindspore import Tensor
from mindspore import nn
from mindspore.ops import operations as P
from mindspore import context
import sys, os
import argparse
from mindspore.train.serialization import save_checkpoint, load_checkpoint

sys.path.append("..")
sys.path.append("../..")
from layer.GNNLayer import SageConv
from src.dataset import load_and_process


class Graphsage(nn.Cell):
    """
    对graphsage网络进行定义
    """
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hops):
        super(Graphsage, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hops = hops
        # 添加Graphsage传播层
        self.layer1 = SageConv(input_dim=self.input_dim,
                               hidden_dim=self.hidden_dim,
                               output_dim=self.hidden_dim,
                               hops=self.hops)
        self.concat = P.Concat(axis=1)

    def construct(self, node_features):
        hidden1 = self.layer1(node_features)
        return hidden1


def train():
    """
    对训练函数进行定义
    """
    # 可设置训练结点个数，后续可把训练参数加入
    parser = argparse.ArgumentParser(description='Graphsage')
    parser.add_argument('--data_dir', type=str, default='../data_mr/cora', help='Dataset directory')
    parser.add_argument('--train_nodes_num', type=int, default=1208, help='Nodes numbers for training')
    parser.add_argument('--eval_nodes_num', type=int, default=500, help='Nodes numbers for evaluation')
    parser.add_argument('--test_nodes_num', type=int, default=1000, help='Nodes numbers for test')
    args = parser.parse_args()

    # 创建文件，保存最优训练模型
    if not os.path.exists("ckpts_graphsage"):
        os.mkdir("ckpts_graphsage")

    # 对模式、环境进行定义
    context.set_context(mode=context.PYNATIVE_MODE,
                        device_target="CPU",
                        save_graphs=False)

    # 读取训练、验证、测试数据
    features, labels, train_mask, test_mask, eval_mask = load_and_process(args.data_dir,
                                                                          args.train_nodes_num,
                                                                          args.eval_nodes_num,
                                                                          args.test_nodes_num)
    rand_incides = np.random.permutation(features.shape[0])
    test_nodes = rand_incides[args.train_nodes_num+args.eval_nodes_num:]
    val_nodes = rand_incides[args.train_nodes_num:args.train_nodes_num+args.eval_nodes_num]
    train_nodes = rand_incides[:args.train_nodes_num]
    feature_size = features.shape[2]
    num_nodes = features.shape[0]
    num_class = labels.max() + 1
    print("feature size: ", feature_size)
    print("nodes number: ", num_nodes)
    print("node classes: ", num_class)

    # 定义训练参数、损失函数、优化器、训练过程
    early_stopping = 15
    eval_acc_max = 0.8
    net_original = Graphsage(input_dim=1433, hidden_dim=128, output_dim=7, hops=[10, 10])
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    opt_Adam = nn.Adam(net_original.trainable_params())
    net_with_loss = nn.WithLossCell(net_original, loss_fn=loss)
    net_train_step = nn.TrainOneStepCell(net_with_loss, opt_Adam)

    for epoch in range(10):
        net_train_step.set_train(mode=True)
        for batch in range(20):
            # 取每一个batch的训练数据
            batch_src_index = np.random.choice(train_nodes, size=(16,))
            features_sampled = []
            for node in batch_src_index:
                features_sampled.append((features[node]))
            batch_train_mask = train_mask[batch_src_index]
            label_source = labels[batch_src_index]
            train_step_loss = net_train_step(Tensor(features_sampled, mindspore.float32),
                                             Tensor(label_source[:, 0], mindspore.int32))
            step_loss = P.ReduceSum()(train_step_loss).asnumpy()

            # 取每一个batch的验证数据
            batch_eval_index = val_nodes
            eval_fea_sampled = []
            for node in batch_eval_index:
                eval_fea_sampled.append((features[node]))
            batch_eval_mask = eval_mask[batch_eval_index]
            eval_label_source = labels[batch_eval_index]
            eval_lable = Tensor(eval_label_source[:, 0], mindspore.int32)
            eval_soln = net_original(Tensor(eval_fea_sampled, mindspore.float32))
            eval_logits = P.Argmax()(eval_soln)
            eval_acc = P.ReduceMean()(P.Cast()((P.Equal()(eval_lable, eval_logits)), mindspore.float32))

            print("Epoch:", epoch + 1, " Batch: ", batch + 1, "'s train loss =", step_loss,
                  " val accuracy =", eval_acc)

            # 保存最优模型
            if eval_acc.asnumpy() > eval_acc_max:
                eval_acc_max = eval_acc
                print("a more accurate model!")
                if os.path.exists("ckpts_graphsage/graphsage.ckpt"):
                    os.remove("ckpts_graphsage/graphsage.ckpt")
                save_checkpoint(net_train_step, "ckpts_graphsage/graphsage.ckpt")

    # 取测试数据
    batch_test_index = test_nodes
    test_fea_sampled = []
    for node in batch_test_index:
        test_fea_sampled.append((features[node]))
    batch_test_mask = eval_mask[batch_test_index]
    test_label_source = labels[batch_test_index]
    test_lable = Tensor(test_label_source[:, 0], mindspore.int32)

    # 读取最优模型，进行测试集上的预测
    test_net = Graphsage(input_dim=1433, hidden_dim=128, output_dim=7, hops=[10, 10])
    test_net.set_train(mode=False)
    load_checkpoint("ckpts_graphsage/graphsage.ckpt", net=test_net)
    loss_test = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    test_soln = test_net(Tensor(test_fea_sampled, mindspore.float32))
    test_logits = P.Argmax()(test_soln)
    print("test accuracy:", P.ReduceMean()(P.Cast()((P.Equal()(test_lable, test_logits)), mindspore.float32)))

    test_with_loss = nn.WithLossCell(test_net, loss_fn=loss_test)
    test_loss = test_with_loss(Tensor(test_fea_sampled, mindspore.float32),
                               Tensor(test_label_source[:, 0], mindspore.int32))
    print("test loss:", P.ReduceSum()(test_loss).asnumpy())


if __name__ == '__main__':
    train()