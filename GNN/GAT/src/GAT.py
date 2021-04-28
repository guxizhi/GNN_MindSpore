import numpy as np
from mindspore import Tensor
from mindspore import nn
from mindspore import context
import argparse
import sys, os
from mindspore.train.serialization import save_checkpoint, load_checkpoint

sys.path.append("..")
sys.path.append("../..")
from layer.GNNLayer import GATConv
from src.wrapper import LossAccuracyWrapper, TrainGAT
from src.dataset import load_and_process


class GAT(nn.Cell):
    """
    对GAT网络进行定义
    """
    def __init__(self,
                input_dim,
                hidden_dim,
                output_dim,
                heads):
        super(GAT, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.heads = heads
        self.input_layer = GATConv(input_dim=self.input_dim,
                                output_dim=self.hidden_dim,
                                heads=self.heads)
        self.output_layer = GATConv(input_dim=self.hidden_dim * self.heads,
                                    output_dim=self.output_dim,
                                    heads=1)

    def construct(self, node_features, biases, training=True):
        hidden = self.input_layer(node_features, biases, training)
        output = self.output_layer(hidden, biases, training)
        return output

def train():
    """
    对训练函数进行定义
    """
    # 可设置训练结点个数
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../../data_mr/ppi', help='Data dir')
    parser.add_argument('--train_nodes_num', type=int, default=44906, help='Nodes numbers for training')
    parser.add_argument('--eval_nodes_num', type=int, default=6514, help='Nodes numbers for evaluation')
    parser.add_argument('--test_nodes_num', type=int, default=5524, help='Nodes numbers for test')
    args = parser.parse_args()

    # 创建文件，保存最有训练模型
    if not os.path.exists("ckpts_gat"):
        os.mkdir("ckpts_gat")

    # 对模式、环境进行定义
    context.set_context(mode=context.PYNATIVE_MODE,
                        device_target="GPU",
                        save_graphs=False)


    # 读取训练、验证、测试数据
    feature, biases, y_train, train_mask, y_val, eval_mask, y_test, test_mask = load_and_process(args.data_dir,
                                                                                                 args.train_nodes_num,
                                                                                                 args.eval_nodes_num,
                                                                                                 args.test_nodes_num)
    feature_size = feature.shape[2]
    num_nodes = feature.shape[1]
    num_class = y_train.shape[2]
    print("feature size: ", feature_size)
    print("nodes number: ", num_nodes)
    print("node classes: ", num_class)

    # 定义训练参数、损失函数、优化器、训练过程
    early_stopping = 100
    lr = 0.005
    l2_coeff = 0.0005
    num_epochs = 200
    val_acc_max = 0.0
    val_loss_min = np.inf
    curr_step = 0
    gat_net = GAT(input_dim=1433, hidden_dim=8, output_dim=7, heads=8)
    gat_net.add_flags_recursive(fp16=True)
    eval_net = LossAccuracyWrapper(gat_net, num_class, y_val, eval_mask, l2_coeff)
    train_net = TrainGAT(gat_net, num_class, y_train, train_mask, lr, l2_coeff)

    feature = Tensor(feature)
    biases = Tensor(biases)


    for _epoch in range(num_epochs):
        # print(feature.shape)
        # 对每一个epoch进行训练，得到loss和训练集acc
        train_net.set_train(True)
        train_result = train_net(feature, biases)
        train_loss = train_result[0].asnumpy()
        train_acc = train_result[1].asnumpy()

        # 对每一次训练得到的模型进行验证，得到loss和验证集acc
        eval_net.set_train(False)
        eval_result = eval_net(feature, biases)
        eval_loss = eval_result[0].asnumpy()
        eval_acc = eval_result[1].asnumpy()

        print("Epoch:{}, train loss={:.5f}, train acc={:.5f} | val loss={:.5f}, val acc={:.5f}".format(
            _epoch, train_loss, train_acc, eval_loss, eval_acc))

        # 保存最优模型
        if eval_acc >= val_acc_max or eval_loss < val_loss_min:
            if eval_acc >= val_acc_max and eval_loss < val_loss_min:
                val_acc_model = eval_acc
                val_loss_model = eval_loss
                if os.path.exists("ckpts_gat/gat.ckpt"):
                    os.remove("ckpts_gat/gat.ckpt")
                save_checkpoint(train_net.network, "ckpts_gat/gat.ckpt")
            val_acc_max = np.max((val_acc_max, eval_acc))
            val_loss_min = np.min((val_loss_min, eval_loss))
            curr_step = 0
        else:
            curr_step += 1
            # 触发提前终止训练条件
            if curr_step == early_stopping:
                print("Early Stop Triggered!, Min loss: {}, Max accuracy: {}".format(val_loss_min, val_acc_max))
                print("Early stop model validation loss: {}, accuracy{}".format(val_loss_model, val_acc_model))
                break

    # 读取最优模型，进行测试集上的预测
    gat_net_test = GAT(input_dim=1433, hidden_dim=8, output_dim=7, heads=8)
    load_checkpoint("ckpts_gat/gat.ckpt", net=gat_net_test)
    gat_net_test.add_flags_recursive(fp16=True)

    test_net = LossAccuracyWrapper(gat_net_test, num_class, y_test, test_mask, l2_coeff)
    test_result = test_net(feature, biases)
    print("Test loss={}, test acc={}".format(test_result[0], test_result[1]))


if __name__ == '__main__':
    train()