import mindspore
import mindspore.dataset as ds
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
from layer.GNNLayer import GINConv
from src.dataset import load_and_process


class GIN(nn.Cell):
    """
    对GIN网络进行定义
    """
    def __init__(self,
                 input_dim=1433,
                 hidden_dim=128,
                 output_dim=7,
                 hops=[10, 10]):
        super(GIN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hops = hops
        # 添加GIN传播层
        self.layer1 = GINConv(input_dim=self.input_dim,
                              hidden_dim=self.hidden_dim,
                              output_dim=self.hidden_dim,
                              hops=self.hops)
        self.concat = P.Concat(axis=1)

    def construct(self, node_features):
        hidden = self.layer1(node_features)
        return hidden

def train():
    """
    对训练函数进行定义
    """
    # 可设置训练结点个数
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument('--data_dir', type=str, default='../data_mr/cora', help='Dataset directory')
    parser.add_argument('--train_nodes_num', type=int, default=1208, help='Nodes numbers for training')
    parser.add_argument('--eval_nodes_num', type=int, default=500, help='Nodes numbers for evaluation')
    parser.add_argument('--test_nodes_num', type=int, default=1000, help='Nodes numbers for test')
    args = parser.parse_args()

    # 创建文件，保存最优训练模型
    if not os.path.exists("ckpts_gin"):
        os.mkdir("ckpts_gin")

    # 对模式、环境进行定义
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", save_graphs=False)

    # 读取训练、验证、测试数据
    features, labels, train_mask, test_mask, eval_mask = load_and_process(args.data_dir,
                                                                          args.train_nodes_num,
                                                                          args.eval_nodes_num,
                                                                          args.test_nodes_num)
    rand_incides = np.random.permutation(features.shape[0])
    test_nodes = rand_incides[args.train_nodes_num+args.eval_nodes_num:]
    val_nodes = rand_incides[args.train_nodes_num:args.train_nodes_num+args.eval_nodes_num]
    train_nodes = rand_incides[:args.train_nodes_num]

    # 定义训练参数
    early_stopping = 15
    eval_acc_max = 0
    max_epoch = 0
    max_batch = 0
    net_original = GIN(input_dim=1433, hidden_dim=128, output_dim=7)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    # opt_momentum = nn.Momentum(net_original.trainable_params(), learning_rate=0.1, momentum=0.9, weight_decay=0.0)
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
                if os.path.exists("ckpts_gin/gin.ckpt"):
                    os.remove("ckpts_gin/gin.ckpt")
                save_checkpoint(net_train_step, "ckpts_gin/gin.ckpt")
                
    # 取测试数据
    batch_test_index = test_nodes
    test_fea_sampled = []
    for node in batch_test_index:
        test_fea_sampled.append((features[node]))
    batch_test_mask = eval_mask[batch_test_index]
    test_label_source = labels[batch_test_index]
    test_lable = Tensor(test_label_source[:, 0], mindspore.int32)

    # 读取最优模型，进行测试集上的预测
    test_net = GIN(input_dim=1433, hidden_dim=128, output_dim=7)
    test_net.set_train(mode=False)
    load_checkpoint("ckpts_gin/gin.ckpt", net=test_net)
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