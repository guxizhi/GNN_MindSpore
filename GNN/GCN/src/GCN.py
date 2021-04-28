import numpy as np
from mindspore import Tensor
from mindspore import nn
from mindspore import context
import argparse
import sys, os
from mindspore.train.serialization import save_checkpoint, load_checkpoint

sys.path.append("..")
sys.path.append("../..")
from layer.GNNLayer import GCNConv
from src.wrapper import LossAccuracyWrapper, TrainNetWrapper
from src.dataset import load_and_process


class GCN(nn.Cell):
    """
    对GCN网络进行定义
    """
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 dropout):
        super(GCN, self).__init__()
        self.input_layer = GCNConv(input_dim, hidden_dim, activation="relu", dropout_ratio=dropout)
        self.output_layer = GCNConv(hidden_dim, output_dim, dropout_ratio=None)

    def construct(self, adj, feature):
        output = self.input_layer(adj, feature)
        output = self.output_layer(adj, output)
        return output


def train():
    """
    对训练函数进行定义
    """
    # 可设置训练结点个数
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument('--data_dir', type=str, default='../data_mr/cora', help='Dataset directory')
    parser.add_argument('--train_nodes_num', type=int, default=140, help='Nodes numbers for training')
    parser.add_argument('--eval_nodes_num', type=int, default=500, help='Nodes numbers for evaluation')
    parser.add_argument('--test_nodes_num', type=int, default=1000, help='Nodes numbers for test')
    args = parser.parse_args()

    # 创建文件，保存最有训练模型
    if not os.path.exists("ckpts_gcn"):
        os.mkdir("ckpts_gcn")

    # 对模式、环境进行定义
    context.set_context(mode=context.PYNATIVE_MODE,
                        device_target="GPU",
                        save_graphs=False)

    # 读取训练、验证、测试数据
    adj, feature, label_onehot, label, train_mask, test_mask, eval_mask= load_and_process(args.data_dir,
                                                                                          args.train_nodes_num,
                                                                                          args.eval_nodes_num,
                                                                                          args.test_nodes_num)
    feature_size = feature.shape[1]
    num_nodes = feature.shape[0]
    class_num = label_onehot.shape[1]
    input_dim = feature.shape[1]
    print("feature size: ", feature_size)
    print("nodes number: ", num_nodes)
    print("node classes: ", class_num)

    # 定义训练参数、损失函数、优化器、训练过程
    learning_rate = 0.01
    epochs = 200
    hidden_dim = 16
    dropout = 0.5
    weight_decay = 5e-4
    early_stopping = 50
    val_acc_max = 0.0
    val_loss_min = np.inf
    curr_step = 0
    gcn_net = GCN(input_dim, hidden_dim, class_num, dropout)
    gcn_net.add_flags_recursive(fp16=True)
    eval_net = LossAccuracyWrapper(gcn_net, label_onehot, eval_mask, weight_decay)
    train_net = TrainNetWrapper(gcn_net, label_onehot, train_mask, weight_decay, learning_rate)

    adj = Tensor(adj)
    feature = Tensor(feature)

    loss_list = []
    for epoch in range(epochs):
        # 对每一个epoch进行训练，得到loss和训练集acc
        train_net.set_train()
        train_result = train_net(adj, feature)
        train_loss = train_result[0].asnumpy()
        train_accuracy = train_result[1].asnumpy()

        # 对每一次训练得到的模型进行验证，得到loss和验证集acc
        eval_net.set_train(False)
        eval_result = eval_net(adj, feature)
        eval_loss = eval_result[0].asnumpy()
        eval_accuracy = eval_result[1].asnumpy()

        loss_list.append(eval_loss)
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss),
              "train_acc=", "{:.5f}".format(train_accuracy), "val_loss=", "{:.5f}".format(eval_loss),
              "val_acc=", "{:.5f}".format(eval_accuracy))

        # 保存最优模型
        if eval_accuracy >= val_acc_max or eval_loss < val_loss_min:
            if eval_accuracy >= val_acc_max and eval_loss < val_loss_min:
                if os.path.exists("ckpts_gcn/gcn.ckpt"):
                    os.remove("ckpts_gcn/gcn.ckpt")
                save_checkpoint(train_net.network, "ckpts_gcn/gcn.ckpt")
            val_acc_max = np.max((val_acc_max, eval_accuracy))
            val_loss_min = np.min((val_loss_min, eval_loss))
            curr_step = 0
        else:
            curr_step += 1
            # 触发提前终止训练条件
            if epoch > early_stopping and loss_list[-1] > np.mean(loss_list[-(early_stopping+1):-1]):
                print("Early stopping...")
                break

    # 读取最优模型，进行测试集上的预测
    gcn_net_test = GCN(input_dim, hidden_dim, class_num, dropout)
    load_checkpoint("ckpts_gcn/gcn.ckpt", net=gcn_net_test)
    gcn_net_test.add_flags_recursive(fp16=True)

    test_net = LossAccuracyWrapper(gcn_net_test, label_onehot, test_mask, weight_decay)
    test_net.set_train(False)
    test_result = test_net(adj, feature)
    test_loss = test_result[0].asnumpy()
    test_accuracy = test_result[1].asnumpy()
    print("Test set results:", "loss=", "{:.5f}".format(test_loss),
          "accuracy=", "{:.5f}".format(test_accuracy))


if __name__ == '__main__':
    train()