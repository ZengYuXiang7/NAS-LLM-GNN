# coding : utf-8
# Author : yuxiang Zeng
import collections
import math
import time

import pickle
import numpy as np
import argparse
import copy

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import *
import torch

from modules.datasets.chatgpt import NAS_ChatGPT
from utils.dataloader import get_dataloaders
from utils.logger import Logger
from utils.metrics import ErrorMetrics
from utils.monitor import EarlyStopping
from utils.trainer import get_loss_function, get_optimizer
from utils.utils import optimizer_zero_grad, optimizer_step, lr_scheduler_step, set_settings, set_seed
import pandas
import re
global log
from utils.utils import *

torch.set_default_dtype(torch.float32)



class experiment:
    def __init__(self, args):
        self.args = args

    # 只是读取大文件
    def load_data(self, args):
        import os
        file_names = os.listdir(args.path + args.dataset)
        pickle_files = [file for file in file_names if file.endswith('.pickle')]
        data = []
        for i in range(len(pickle_files)):
            pickle_file = args.path + pickle_files[i]
            with open(pickle_file, 'rb') as f:
                now = pickle.load(f)
            data.append([now])
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
            for i in range(len(data)):
                for key in (data[i][0].keys()):
                    now = []
                    # 添加设备号
                    now.append(i)
                    matrix, features = self.get_graph(key)
                    # print(np.array(matrix).shape)
                    # print(np.array(features).shape)
                    now.append(np.array(matrix))
                    now.append(np.array(features))
                    y = data[i][0][key]
                    now.append(y)
                    tensor.append(now)
                    # print(now)
            tensor = np.array(tensor)
        return tensor

    def get_pytorch_index(self, data):
        return data

    def get_matrix_and_ops(self, g, prune=True, keep_dims=True):
        ''' Return the adjacency matrix and label vector.

            Args:
                g : should be a point from Nasbench102 search space
                prune : remove dangling nodes that only connected to zero ops
                keep_dims : keep the original matrix size after pruning
        '''

        matrix = [[0 for _ in range(8)] for _ in range(8)]
        labels = [None for _ in range(8)]
        labels[0] = 'input'
        labels[-1] = 'output'
        matrix[0][1] = matrix[0][2] = matrix[0][4] = 1
        matrix[1][3] = matrix[1][5] = 1
        matrix[2][6] = 1
        matrix[3][6] = 1
        matrix[4][7] = 1
        matrix[5][7] = 1
        matrix[6][7] = 1

        for idx, op in enumerate(g):
            if op == 0:  # zero
                for other in range(8):
                    if matrix[other][idx + 1]:
                        matrix[other][idx + 1] = 0
                    if matrix[idx + 1][other]:
                        matrix[idx + 1][other] = 0
            elif op == 1:  # skip-connection:
                to_del = []
                for other in range(8):
                    if matrix[other][idx + 1]:
                        for other2 in range(8):
                            if matrix[idx + 1][other2]:
                                matrix[other][other2] = 1
                                matrix[other][idx + 1] = 0
                                to_del.append(other2)
                for d in to_del:
                    matrix[idx + 1][d] = 0
            else:
                labels[idx + 1] = str(op)

        if prune:
            visited_fw = [False for _ in range(8)]
            visited_bw = copy.copy(visited_fw)

            def bfs(beg, vis, con_f):
                q = [beg]
                vis[beg] = True
                while q:
                    v = q.pop()
                    for other in range(8):
                        if not vis[other] and con_f(v, other):
                            q.append(other)
                            vis[other] = True

            bfs(0, visited_fw, lambda src, dst: matrix[src][dst])  # forward
            bfs(7, visited_bw, lambda src, dst: matrix[dst][src])  # backward

            for v in range(7, -1, -1):
                if not visited_fw[v] or not visited_bw[v]:
                    labels[v] = None
                    if keep_dims:
                        matrix[v] = [0] * 8
                    else:
                        del matrix[v]
                    for other in range(len(matrix)):
                        if keep_dims:
                            matrix[other][v] = 0
                        else:
                            del matrix[other][v]

            if not keep_dims:
                labels = list(filter(lambda l: l is not None, labels))

            assert visited_fw[-1] == visited_bw[0]
            assert visited_fw[-1] == False or matrix

            verts = len(matrix)
            assert verts == len(labels)
            for row in matrix:
                assert len(row) == verts

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


# 数据集定义
class DataModule:
    def __init__(self, exper_type, args):
        self.args = args
        self.path = args.path
        self.data = exper_type.load_data(args)
        self.data = exper_type.preprocess_data(self.data, args)
        self.train_tensor, self.valid_tensor, self.test_tensor, self.max_value = self.get_train_valid_test_dataset(self.data, args)
        self.train_set, self.valid_set, self.test_set = self.get_dataset(self.train_tensor, self.valid_tensor, self.test_tensor, exper_type, args)
        self.train_loader, self.valid_loader, self.test_loader = get_dataloaders(self.train_set, self.valid_set, self.test_set, args)
        args.log.only_print(f'Train_length : {len(self.train_tensor)} Valid_length : {len(self.valid_tensor)} Test_length : {len(self.test_tensor)}')

    def get_train_valid_test_dataset(self, tensor, args):
        p = np.random.permutation(len(tensor))
        tensor = tensor[p]

        X = tensor[:, :-1]
        Y = tensor[:, -1].reshape(-1, 1)
        # max_value = Y.max()
        max_value = 1
        Y /= max_value

        train_size = int(len(tensor) * args.density)
        if args.dataset == 'cpu':
            valid_size = int(100)
        elif args.dataset == 'gpu':
            valid_size = int(200)

        X_train = X[:train_size]
        Y_train = Y[:train_size]

        X_valid = X[train_size:train_size + valid_size]
        Y_valid = Y[train_size:train_size + valid_size]

        X_test = X[train_size + valid_size:]
        Y_test = Y[train_size + valid_size:]

        train_tensor = np.hstack((X_train, Y_train))
        valid_tensor = np.hstack((X_valid, Y_valid))
        test_tensor = np.hstack((X_test, Y_test))

        return train_tensor, valid_tensor, test_tensor, max_value

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

    def __getitem__(self, idx):
        device_idx = self.indices[idx][0]
        matrix = self.indices[idx][1]
        features = self.indices[idx][2]
        value = self.indices[idx, -1]
        return device_idx, matrix, features, value

    def __len__(self):
        return self.indices.shape[0]



class ARNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(ARNN, self).__init__()
        self.rnn = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)

    def forward(self, x, adj_matrix):
        # x: 节点特征矩阵，形状为 (batch_size, num_nodes, feature_size)
        # adj_matrix: 批处理的图的邻接矩阵，形状为 (batch_size, num_nodes, num_nodes)
        batch_size, num_nodes, _ = x.shape
        x = x.to(torch.float32)
        adj_matrix = adj_matrix.to(torch.float32)

        updated_features_batch = []

        # 处理每个样本
        for b in range(batch_size):
            updated_features = []
            for i in range(num_nodes):  # 遍历所有节点
                neighbors_indices = (adj_matrix[b, i] > 0).nonzero(as_tuple=False).view(-1)
                node_features = x[b, i, :]  # 当前节点特征
                neighbor_features = [node_features.unsqueeze(0)]  # 包括节点自身，增加一个维度以匹配

                for neighbor_index in neighbors_indices:
                    neighbor_features.append(x[b, neighbor_index, :].unsqueeze(0))  # 添加邻居节点特征

                # 计算平均特征向量
                neighbor_features = torch.cat(neighbor_features, dim=0)
                avg_feature = torch.mean(neighbor_features, dim=0, keepdim=True)
                updated_features.append(avg_feature.squeeze(0))  # 移除多余的维度

            # 将更新后的特征向量堆叠为一个新的特征矩阵
            updated_features_batch.append(torch.stack(updated_features, dim=0))

        updated_features_batch = torch.stack(updated_features_batch, dim=0)
        updated_features_batch = updated_features_batch.float()

        # 经过RNN
        out, (hn, cn) = self.rnn(updated_features_batch)

        # 这里是处理双向LSTM的逻辑，根据你的LSTM配置可能需要调整
        hn_fwd = hn[-2, :, :]  # 前向的最后隐藏状态
        hn_bwd = hn[-1, :, :]  # 后向的最后隐藏状态
        hn_combined = torch.cat((hn_fwd, hn_bwd), dim=1)  # 形状: (batch_size, hidden_dim * 2)

        return hn_combined


class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.input_dim = 6
        self.hidden_dim = args.dimension
        self.arnn = ARNN(self.input_dim, self.hidden_dim, 1)
        self.fc = torch.nn.Linear(self.hidden_dim * 2 + 1, 1)

    def forward(self, device_idx, matrix, features):
        dnn_embeds = self.arnn(features, matrix)
        # print(dnn_embeds.shape)
        # print(device_idx.shape)
        device_idx = device_idx.unsqueeze(1)  # 形状: (batch_size, 1)
        final_inputs = torch.cat([device_idx, dnn_embeds], dim=1)  # 形状: (batch_size, hidden_dim * 2 + 1)
        y = self.fc(final_inputs)
        return y.flatten()

    def setup_optimizer(self, args):
        self.to(args.device)
        self.loss_function = get_loss_function(args).to(args.device)
        self.optimizer = get_optimizer(self.parameters(), lr=args.lr, decay=args.decay, args=args)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, threshold=0.01)

    def set_epochs(self, epochs):
        self.epochs = epochs

    def train_one_epoch(self, dataModule):
        loss = None
        self.train()
        torch.set_grad_enabled(True)
        t1 = time.time()
        for train_Batch in dataModule.train_loader:
            device_idx, matrix, features, value = train_Batch
            pred = self.forward(device_idx, matrix, features)
            loss = self.loss_function(pred, value)
            optimizer_zero_grad(self.optimizer)
            loss.backward()
            optimizer_step(self.optimizer)
        t2 = time.time()
        self.eval()
        torch.set_grad_enabled(False)
        return loss, t2 - t1

    def valid_one_epoch(self, dataModule):
        writeIdx = 0
        val_loss = 0.
        preds = torch.zeros((len(dataModule.valid_loader.dataset),)).to(self.args.device)
        reals = torch.zeros((len(dataModule.valid_loader.dataset),)).to(self.args.device)
        for valid_Batch in dataModule.valid_loader:
            device_idx, matrix, features, value = valid_Batch
            pred = self.forward(device_idx, matrix, features)
            val_loss += self.loss_function(pred, value).item()
            preds[writeIdx:writeIdx + len(pred)] = pred
            reals[writeIdx:writeIdx + len(value)] = value
            writeIdx += len(pred)
        self.scheduler.step(val_loss)
        valid_error = ErrorMetrics(reals * dataModule.max_value, preds * dataModule.max_value)
        return valid_error

    def test_one_epoch(self, dataModule):
        writeIdx = 0
        preds = torch.zeros((len(dataModule.test_loader.dataset),)).to(self.args.device)
        reals = torch.zeros((len(dataModule.test_loader.dataset),)).to(self.args.device)
        for test_Batch in dataModule.test_loader:
            device_idx, matrix, features, value = test_Batch
            pred = self.forward(device_idx, matrix, features)
            preds[writeIdx:writeIdx + len(pred)] = pred
            reals[writeIdx:writeIdx + len(value)] = value
            writeIdx += len(pred)
        test_error = ErrorMetrics(reals * dataModule.max_value, preds * dataModule.max_value)
        return test_error


def get_dataloaders(train_set, valid_set, test_set, args):
    # max_workers = multiprocessing.cpu_count()
    # max_workers = 1

    train_loader = DataLoader(
        train_set,
        batch_size=args.bs,
        drop_last=False,
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=1024,
        drop_last=False,
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1024,  # 8192
        drop_last=False,
        shuffle=False,
        collate_fn=custom_collate_fn
    )

    return train_loader, valid_loader, test_loader



def custom_collate_fn(batch):
    from torch.utils.data.dataloader import default_collate
    return default_collate(batch)


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
    for epoch in trange(args.epochs, disable=not args.program_test):
        model.set_epochs(epoch)
        epoch_loss, time_cost = model.train_one_epoch(datamodule)
        valid_error = model.valid_one_epoch(datamodule)
        monitor.track(epoch, model.state_dict(), valid_error['MAE'])
        train_time.append(time_cost)
        if args.verbose and epoch % args.verbose == 0 and not args.program_test:
            log.only_print(f"Round={runId + 1} Epoch={epoch + 1:02d} Loss={epoch_loss:.4f} vMAE={valid_error['MAE']:.4f} vRMSE={valid_error['RMSE']:.4f} vNMAE={valid_error['NMAE']:.4f} vNRMSE={valid_error['NRMSE']:.4f} time={sum(train_time):.1f} s")
            log.only_print(f"Acc = [1%={valid_error['Acc'][0]:.4f}, 5%={valid_error['Acc'][1]:.4f}, 10%={valid_error['Acc'][2]:.4f}]")
        if monitor.early_stop:
            break
    model.load_state_dict(monitor.best_model)
    sum_time = sum(train_time[: monitor.best_epoch])
    results = model.test_one_epoch(datamodule)
    log(f'Round={runId + 1} BestEpoch={monitor.best_epoch:d} MAE={results["MAE"]:.4f} RMSE={results["RMSE"]:.4f} NMAE={results["NMAE"]:.4f} NRMSE={results["NRMSE"]:.4f} Training_time={sum_time:.1f} s')
    log(f"Acc = [1%={results['Acc'][0]:.4f}, 5%={results['Acc'][1]:.4f}, 10%={results['Acc'][2]:.4f}] ")
    return {
        'MAE': results["MAE"],
        'RMSE': results["RMSE"],
        'NMAE': results["NMAE"],
        'NRMSE': results["NRMSE"],
        'TIME': sum_time,
    }, results['Acc']


def RunExperiments(log, args):
    log('*' * 20 + 'Experiment Start' + '*' * 20)
    metrics = collections.defaultdict(list)
    for runId in range(args.rounds):
        runHash = int(time.time())
        results, acc = RunOnce(args, runId, runHash, log)
        for key in results:
            metrics[key].append(results[key])
        for key, item in zip(['Acc1', 'Acc5', 'Acc10'], [0, 1, 2]):
            metrics[key].append(acc[item])
    log('*' * 20 + 'Experiment Results:' + '*' * 20)
    for key in metrics:
        log(f'{key}: {np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f}')
    if args.record:
        log.save_result(metrics)
    log('*' * 20 + 'Experiment Success' + '*' * 20 + '\n')
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rounds', type=int, default=5)

    parser.add_argument('--dataset', type=str, default='gpu')  #
    parser.add_argument('--model', type=str, default='LSTM')  #

    # Experiment
    parser.add_argument('--density', type=float, default=0.05)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--record', type=int, default=1)
    parser.add_argument('--program_test', type=int, default=1)
    parser.add_argument('--experiment', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--path', nargs='?', default='./datasets/')

    # Training tool
    parser.add_argument('--device', type=str, default='cpu')  # gpu cpu mps
    parser.add_argument('--bs', type=int, default=1)  #
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--decay', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=500)
    parser.add_argument('--saved', type=int, default=1)

    parser.add_argument('--loss_func', type=str, default='L1Loss')
    parser.add_argument('--optim', type=str, default='AdamW')

    # Hyper parameters
    parser.add_argument('--dimension', type=int, default=32)

    # Other Experiment
    parser.add_argument('--ablation', type=int, default=0)
    parser.add_argument('--llm', type=int, default=1)
    args = parser.parse_args()
    set_settings(args)
    log = Logger(args)
    args.log = log
    log(str(args))
    RunExperiments(log, args)



