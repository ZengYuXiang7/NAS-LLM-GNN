# coding : utf-8
# Author : yuxiang Zeng
import collections
import time

import pickle
import numpy as np
import argparse
import copy

from sklearn.model_selection import train_test_split
from tqdm import *
import torch
from utils.dataloader import get_dataloaders
from utils.logger import Logger
from utils.metrics import ErrorMetrics
from utils.monitor import EarlyStopping
from utils.trainer import get_loss_function, get_optimizer
from utils.utils import optimizer_zero_grad, optimizer_step, lr_scheduler_step, set_settings, set_seed

global log


class experiment:
    def __init__(self, args):
        self.args = args

    # 只是读取大文件
    def load_data(self, args):
        import os
        file_names = os.listdir(args.path)
        pickle_files = [file for file in file_names if file.endswith('.pickle')]
        data = []
        for i in range(len(pickle_files)):
            pickle_file = args.path + pickle_files[i]
            with open(pickle_file, 'rb') as f:
                now = pickle.load(f)
            data.append([now])
            break
        data = np.array(data)
        return data

    def get_graph(self, op_seq):
        op_seq = list(op_seq)
        matrix, label = self.get_matrix_and_ops(op_seq)
        matrix, features = self.get_adjacency_and_features(matrix, label)
        return matrix, features

    def preprocess_data(self, data, args):
        try:
            tensor = pickle.load(open(f'./pretrained/tensor_{torch.initial_seed()}.pkl', 'rb'))
        except:
            tensor = []
            for i in trange(len(data)):
                for key in (data[i][0].keys()):
                    now = []
                    # 添加设备号
                    matrix, features = self.get_graph(key)
                    now.append(np.array(matrix))
                    now.append(np.array(features))
                    y = data[i][0][key]
                    now.append(y)
                    tensor.append(now)
            tensor = np.array(tensor)
        return tensor

    def get_pytorch_index(self, data):
        return data


    def get_arch_vector_from_arch_str(self, arch_str):
        """
            Args:
                arch_str : a string representation of a cell architecture,
                    for example '|nor_conv_3x3~0|+|nor_conv_3x3~0|avg_pool_3x3~1|+|skip_cotorch.nnect~0|nor_conv_3x3~1|skip_cotorch.nnect~2|'
        """
        _opname_to_index = {
            'none': 0,
            'skip_cotorch.nnect': 1,
            'nor_conv_1x1': 2,
            'nor_conv_3x3': 3,
            'avg_pool_3x3': 4,
            'input': 5,
            'output': 6,
            'global': 7
        }

        _opindex_to_name = {value: key for key, value in _opname_to_index.items()}
        nodes = arch_str.split('+')
        nodes = [node[1:-1].split('|') for node in nodes]
        nodes = [[op_and_input.split('~')[0] for op_and_input in node] for node in nodes]

        # arch_vector is equivalent to a decision vector produced by autocaml when using Nasbench201 backend
        arch_vector = [_opname_to_index[op] for node in nodes for op in node]
        return arch_vector


    def get_arch_str_from_arch_vector(self, arch_vector):
        _opname_to_index = {
            'none': 0,
            'skip_cotorch.nnect': 1,
            'nor_conv_1x1': 2,
            'nor_conv_3x3': 3,
            'avg_pool_3x3': 4,
            'input': 5,
            'output': 6,
            'global': 7
        }
        _opindex_to_name = {value: key for key, value in _opname_to_index.items()}
        ops = [_opindex_to_name[opindex] for opindex in arch_vector]
        return '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.format(*ops)


    def get_matrix_and_ops(self, g, prune=True, keep_dims=True):
        """
        返回邻接矩阵和节点标签。
        Args:
            g : 来自 Nasbench102 搜索空间的一个点（图结构）（列表形式）
            prune : 是否删除只连接到零操作的悬挂节点，默认为 True
            keep_dims : 在剪枝后是否保持原始矩阵大小，默认为 False
        """
        # 初始化8x8的邻接矩阵，所有元素都设为0
        matrix = [[0 for _ in range(8)] for _ in range(8)]

        # 初始化节点标签列表，初始时所有节点标签都为 None
        labels = [None for _ in range(8)]

        # 设置输入节点和输出节点的标签
        labels[0] = 'input'
        labels[-1] = 'output'

        # 初始化部分邻接矩阵的连接关系
        matrix[0][1] = matrix[0][2] = matrix[0][4] = 1
        matrix[1][3] = matrix[1][5] = 1
        matrix[2][6] = 1
        matrix[3][6] = 1
        matrix[4][7] = 1
        matrix[5][7] = 1
        matrix[6][7] = 1

        # 遍历输入的图结构
        for idx, op in enumerate(g):
            # 如果操作是零
            if op == 0:
                # 删除与该节点连接的边
                for other in range(8):  # 将这该序列点与全部节点断边
                    # print(other, idx + 1)
                    if matrix[other][idx + 1]:
                        matrix[other][idx + 1] = 0
                    if matrix[idx + 1][other]:
                        matrix[idx + 1][other] = 0
            # 如果操作是跳跃连接
            elif op == 1:
                # 处理跳跃连接的情况
                to_del = []
                for other in range(8):
                    # 若有谁连接到我，定点到这个谁
                    if matrix[other][idx + 1]:
                        for other2 in range(8):
                            # 若我也连接到别人
                            if matrix[idx + 1][other2]:
                                # 就把连接到我的直接连到我连接的下一位
                                matrix[other][other2] = 1
                                # 断掉别人连向我的边
                                matrix[other][idx + 1] = 0
                                to_del.append(other2)

                # 删掉这些边
                for d in to_del:
                    matrix[idx + 1][d] = 0
            # 如果操作是其他数字
            else:
                # 设置节点标签为操作的字符串表示
                labels[idx + 1] = str(op)

        # 如果选择剪枝操作
        if prune:
            # 初始化前向和后向访问标记列表
            visited_fw = [False for _ in range(8)]
            visited_bw = copy.copy(visited_fw)

            # 定义广度优先搜索函数
            def bfs(beg, vis, con_f):
                q = [beg]
                vis[beg] = True
                while q:
                    v = q.pop()
                    for other in range(8):
                        if not vis[other] and con_f(v, other):
                            q.append(other)
                            vis[other] = True

            # 前向搜索
            bfs(0, visited_fw, lambda src, dst: matrix[src][dst])
            # 后向搜索
            bfs(7, visited_bw, lambda src, dst: matrix[dst][src])

            # 遍历节点，删除未访问到的节点
            for v in range(7, -1, -1):
                if not visited_fw[v] or not visited_bw[v]:
                    labels[v] = None
                    # 如果选择保持原始矩阵大小
                    if keep_dims:
                        matrix[v] = [0] * 8
                    else:
                        del matrix[v]
                    for other in range(len(matrix)):
                        # 如果选择保持原始矩阵大小
                        if keep_dims:
                            matrix[other][v] = 0
                        else:
                            del matrix[other][v]

            # 如果不保持原始矩阵大小
            if not keep_dims:
                # 移除标签中的 None 元素
                labels = list(filter(lambda l: l is not None, labels))

            # 断言保证首尾节点被访问到
            assert visited_fw[-1] == visited_bw[0]
            # 断言保证至少存在一个连接或者矩阵非空
            assert visited_fw[-1] == False or matrix

            # 断言确保矩阵的维度正确
            verts = len(matrix)
            assert verts == len(labels)
            for row in matrix:
                assert len(row) == verts

        # 返回最终邻接矩阵和节点标签
        return matrix, labels

    def get_adjacency_and_features(self, matrix, labels):
        # Add global node
        for row in matrix:
            row.insert(0, 0)
        global_row = [0, 1, 1, 1, 1, 1, 1, 1, 1]
        matrix.insert(0, global_row)
        # Add diag matrix
        for idx, row in enumerate(matrix):
            row[idx] = 1
        # Create features matrix from labels
        features = [[0 for _ in range(6)] for _ in range(9)]
        features[0][5] = 1  # global
        features[1][3] = 1  # input
        features[-1][4] = 1  # output
        for idx, op in enumerate(labels):
            if op != None and op != 'input' and op != 'output':
                features[idx + 1][int(op) - 2] = 1
        return matrix, features


def get_train_valid_test_dataset(tensor, args):
    X = tensor[:, :-1]
    Y = tensor[:, -1].reshape(-1, 1)
    max_value = Y.max()
    Y /= max_value
    train_size = 900 / len(X)  # 1000是训练集和验证集总和的样本数
    valid_size = 100 / (len(X) - 900)  # 测试集占测试集和验证集总和的比例
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, train_size=train_size)
    X_valid, X_test, Y_valid, Y_test = train_test_split(X_temp, Y_temp, train_size=valid_size)
    train_tensor = np.hstack((X_train, Y_train))
    valid_tensor = np.hstack((X_valid, Y_valid))
    test_tensor = np.hstack((X_test, Y_test))
    return train_tensor, valid_tensor, test_tensor, max_value


# 数据集定义
class DataModule:
    def __init__(self, exper_type, args):
        self.args = args
        self.path = args.path
        self.data = exper_type.load_data(args)
        self.data = exper_type.preprocess_data(self.data, args)
        self.train_tensor, self.valid_tensor, self.test_tensor, self.max_value = get_train_valid_test_dataset(self.data, args)
        self.train_set, self.valid_set, self.test_set = self.get_dataset(self.train_tensor, self.valid_tensor, self.test_tensor, exper_type, args)
        self.train_loader, self.valid_loader, self.test_loader = get_dataloaders(self.train_set, self.valid_set, self.test_set, args)
        args.log.only_print(f'Train_length : {len(self.train_loader) * args.bs} Valid_length : {len(self.valid_loader) * args.bs * 16} Test_length : {len(self.test_loader) * args.bs * 16}')

    def get_dataset(self, train_tensor, valid_tensor, test_tensor, exper_type, args):
        return (
            TensorDataset(train_tensor, exper_type, args),
            TensorDataset(valid_tensor, exper_type, args),
            TensorDataset(test_tensor, exper_type, args)
        )


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensor, exper_type, args):
        self.args = args
        self.tensor = tensor
        self.indices = exper_type.get_pytorch_index(tensor)
        # self.indices = self.delete_zero_row(self.indices)

    def __getitem__(self, idx):
        matrix, features = self.indices[idx, 0], self.indices[idx, 1]
        matrix, features = torch.as_tensor(matrix), torch.as_tensor(features)
        value = torch.as_tensor(self.indices[idx, -1])  # 最后一列作为真实值
        return matrix, features, value

    def __len__(self):
        return self.indices.shape[0]

    def delete_zero_row(self, tensor):
        row_sums = tensor.sum(axis=1)
        nonzero_rows = (row_sums != 0).nonzero().squeeze()
        filtered_tensor = tensor[nonzero_rows]
        return filtered_tensor


class GraphConvolution(torch.torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, weight_init='thomas', bias_init='thomas'):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = torch.torch.nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.weight_init = weight_init
        self.bias_init = bias_init
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, adjacency, features):
        support = torch.matmul(features, self.weight)
        output = torch.bmm(adjacency, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class Model(torch.torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        num_features = 6
        num_layers = 4
        num_hidden = 600
        dropout_ratio = 0
        weight_init = 'thomas'
        bias_init = 'thomas'
        binary_classifier = False
        augments = 0

        self.nfeat = num_features
        self.nlayer = num_layers
        self.nhid = num_hidden
        self.dropout_ratio = dropout_ratio

        self.gc = torch.nn.ModuleList([GraphConvolution(self.nfeat if i == 0 else self.nhid, self.nhid, bias=True, weight_init=weight_init, bias_init=bias_init) for i in range(self.nlayer)])
        self.bn = torch.nn.ModuleList([torch.nn.LayerNorm(self.nhid).float() for i in range(self.nlayer)])
        self.relu = torch.nn.ModuleList([torch.nn.ReLU().float() for i in range(self.nlayer)])
        if not binary_classifier:
            self.fc = torch.nn.Linear(self.nhid + augments, 1).float()
        else:
            if binary_classifier == 'naive':
                self.fc = torch.nn.Linear(self.nhid + augments, 1).float()
            elif binary_classifier == 'oneway' or binary_classifier == 'oneway-hard':
                self.fc = torch.nn.Linear((self.nhid + augments) * 2, 1).float()
            else:
                self.fc = torch.nn.Linear((self.nhid + augments) * 2, 2).float()

            if binary_classifier != 'oneway' and binary_classifier != 'oneway-hard':
                self.final_act = torch.nn.LogSoftmax(dim=1)
            else:
                self.final_act = torch.nn.Sigmoid()
        self.dropout = torch.nn.ModuleList([torch.nn.Dropout(self.dropout_ratio).float() for i in range(self.nlayer)])
        self.binary_classifier = binary_classifier

    def forward_single_model(self, adjacency, features):
        x = self.relu[0](self.bn[0](self.gc[0](adjacency, features)))
        x = self.dropout[0](x)
        for i in range(1, self.nlayer):
            x = self.relu[i](self.bn[i](self.gc[i](adjacency, x)))
            x = self.dropout[i](x)

        return x

    def extract_features(self, adjacency, features, augments=None):
        x = self.forward_single_model(adjacency, features)
        x = x[:, 0]  # use global node
        if augments is not None:
            x = torch.cat([x, augments], dim=1)
        return x

    def regress(self, features, features2=None):
        if not self.binary_classifier:
            assert features2 is None
            return self.fc(features)

        assert features2 is not None
        if self.binary_classifier == 'naive':
            x1 = self.fc(features)
            x2 = self.fc(features2)
        else:
            x1 = features
            x2 = features2

        x = torch.cat([x1, x2], dim=1)
        if self.binary_classifier != 'naive':
            x = self.fc(x)

        x = self.final_act(x)
        return x

    def forward(self, adjacency, features):
        adjacency = adjacency.to(dtype=torch.float)
        features = features.to(dtype=torch.float)
        augments = None
        if not self.binary_classifier:
            x = self.forward_single_model(adjacency, features)
            x = x[:, 0]  # use global node
            if augments is not None:
                x = torch.cat([x, augments], dim=1)
            y = self.fc(x).flatten()
            return y
        else:
            x1 = self.forward_single_model(adjacency[:, 0], features[:, 0])
            x1 = x1[:, 0]
            x2 = self.forward_single_model(adjacency[:, 1], features[:, 1])
            x2 = x2[:, 0]
            if augments is not None:
                a1 = augments[:, 0]
                a2 = augments[:, 1]
                x1 = torch.cat([x1, a1], dim=1)
                x2 = torch.cat([x2, a2], dim=1)

            if self.binary_classifier == 'naive':
                x1 = self.fc(x1)
                x2 = self.fc(x2)

            x = torch.cat([x1, x2], dim=1)
            if self.binary_classifier != 'naive':
                x = self.fc(x)

            x = self.final_act(x)
            return x

    def reset_last(self):
        self.fc.reset_parameters()

    def final_params(self):
        return self.fc.parameters()

    def setup_optimizer(self, args):
        self.to(args.device)
        self.loss_function = get_loss_function(args).to(args.device)
        self.optimizer = get_optimizer(self.parameters(), lr=args.lr, decay=args.decay, args=args)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_step, gamma=0.50)

    def train_one_epoch(self, dataModule):
        loss = None
        self.train()
        torch.set_grad_enabled(True)
        t1 = time.time()
        for train_Batch in tqdm(dataModule.train_loader, disable=not self.args.program_test):
            adjmatrix, features, value = train_Batch
            pred = self.forward(adjmatrix, features)
            loss = self.loss_function(pred.to(torch.float32), value.to(torch.float32))
            optimizer_zero_grad(self.optimizer)
            loss.backward()
            optimizer_step(self.optimizer)
        t2 = time.time()
        self.eval()
        torch.set_grad_enabled(False)
        lr_scheduler_step(self.scheduler)
        return loss, t2 - t1

    def valid_one_epoch(self, dataModule):
        writeIdx = 0
        preds = torch.zeros((len(dataModule.valid_loader.dataset),)).to(self.args.device)
        reals = torch.zeros((len(dataModule.valid_loader.dataset),)).to(self.args.device)
        for valid_Batch in tqdm(dataModule.valid_loader, disable=not self.args.program_test):
            adjmatrix, features, value = valid_Batch
            pred = self.forward(adjmatrix, features)
            preds[writeIdx:writeIdx + len(pred)] = pred
            reals[writeIdx:writeIdx + len(value)] = value
            writeIdx += len(pred)
        valid_error = ErrorMetrics(reals * dataModule.max_value, preds * dataModule.max_value)
        return valid_error

    def test_one_epoch(self, dataModule):
        writeIdx = 0
        preds = torch.zeros((len(dataModule.test_loader.dataset),)).to(self.args.device)
        reals = torch.zeros((len(dataModule.test_loader.dataset),)).to(self.args.device)
        for test_Batch in tqdm(dataModule.test_loader, disable=not self.args.program_test):
            adjmatrix, features, value = test_Batch
            pred = self.forward(adjmatrix, features)
            preds[writeIdx:writeIdx + len(pred)] = pred
            reals[writeIdx:writeIdx + len(value)] = value
            writeIdx += len(pred)
        test_error = ErrorMetrics(reals * dataModule.max_value, preds * dataModule.max_value)
        return test_error


def RunOnce(args, runId, Runtime, log):
    # Set seed
    set_seed(args.seed + runId)

    # Initialize
    exper = experiment(args)
    datamodule = DataModule(exper, args)
    model = Model(args)
    monitor = EarlyStopping(args.patience)

    # Setup training tool
    model.setup_optimizer(args)
    model.max_value = datamodule.max_value
    train_time = []
    for epoch in range(args.epochs):
        epoch_loss, time_cost = model.train_one_epoch(datamodule)
        valid_error = model.valid_one_epoch(datamodule)
        monitor.track(epoch, model.state_dict(), valid_error['MAE'])
        train_time.append(time_cost)
        if args.verbose and epoch % args.verbose == 0:
            log.only_print(
                f"Round={runId + 1} Epoch={epoch + 1:02d} Loss={epoch_loss:.4f} vMAE={valid_error['MAE']:.4f} vRMSE={valid_error['RMSE']:.4f} vNMAE={valid_error['NMAE']:.4f} vNRMSE={valid_error['NRMSE']:.4f}, vAcc={valid_error['Acc']:.4f} time={sum(train_time):.1f} s")
        if monitor.early_stop:
            break
    model.load_state_dict(monitor.best_model)
    sum_time = sum(train_time[: monitor.best_epoch])
    results = model.test_one_epoch(datamodule) if args.valid else valid_error
    log(f'Round={runId + 1} BestEpoch={monitor.best_epoch:d} MAE={results["MAE"]:.4f} RMSE={results["RMSE"]:.4f} NMAE={results["NMAE"]:.4f} NRMSE={results["NRMSE"]:.4f} Acc={results["Acc"]:.4f} Training_time={sum_time:.1f} s\n')
    return {
        'MAE': results["MAE"],
        'RMSE': results["RMSE"],
        'NMAE': results["NMAE"],
        'NRMSE': results["NRMSE"],
        'Acc' : results["Acc"],
        'TIME': sum_time,
    }


def RunExperiments(log, args):
    log('*' * 20 + 'Experiment Start' + '*' * 20)
    metrics = collections.defaultdict(list)

    for runId in range(args.rounds):
        runHash = int(time.time())
        results = RunOnce(args, runId, runHash, log)
        for key in results:
            metrics[key].append(results[key])

    log('*' * 20 + 'Experiment Results:' + '*' * 20)

    for key in metrics:
        log(f'{key}: {np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f}')

    if args.record:
        log.save_result(metrics)

    log('*' * 20 + 'Experiment Success' + '*' * 20)

    return metrics


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rounds', type=int, default=5)

    parser.add_argument('--dataset', type=str, default='rt')  #
    parser.add_argument('--model', type=str, default='CF')  #

    # Experiment
    parser.add_argument('--density', type=float, default=0.10)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--record', type=int, default=1)
    parser.add_argument('--program_test', type=int, default=0)
    parser.add_argument('--valid', type=int, default=1)
    parser.add_argument('--experiment', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=10)
    parser.add_argument('--path', nargs='?', default='./datasets/')

    # Training tool
    parser.add_argument('--device', type=str, default='cpu')  # gpu cpu mps
    parser.add_argument('--bs', type=int, default=16)  #
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--decay', type=float, default=1e-3)
    parser.add_argument('--lr_step', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--saved', type=int, default=1)

    parser.add_argument('--loss_func', type=str, default='L1Loss')
    parser.add_argument('--optim', type=str, default='AdamW')

    # Hyper parameters
    parser.add_argument('--dimension', type=int, default=32)

    # Other Experiment
    parser.add_argument('--ablation', type=int, default=0)
    args = parser.parse_args([])
    return args


if __name__ == '__main__':
    args = get_args()
    set_settings(args)
    log = Logger(args)
    args.log = log
    log(str(args))
    RunExperiments(log, args)



