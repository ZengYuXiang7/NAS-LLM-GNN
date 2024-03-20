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

torch.set_default_dtype(torch.double)



class experiment:
    def __init__(self, args):
        self.args = args
        self.chatgpt = NAS_ChatGPT(args)

    def get_cpu(self, string):
        print(string)
        pattern = re.compile(
            r'^(?P<frequency>[\d.]+)\sGHz::(?P<cores>\d+)\scores::(?P<threads>\d+)\sThreads::(?P<memory_size>[\d.]+)\sMB::(?P<memory_speed>[\d.]+)\sGB/s$')
        match = pattern.match(string)
        if match:
            frequency = float(match.group('frequency'))
            cores = int(match.group('cores'))
            threads = int(match.group('threads'))
            memory_size = float(match.group('memory_size'))
            memory_speed = float(match.group('memory_speed'))
        return [frequency, cores, threads, memory_size, memory_speed]

    def get_gpu(self, string):
        # 3584::1480 MHz::11 GB::352-bit
        print(string)
        pattern = re.compile(
            r'^(?P<Stream_processor_count>\d+)::(?P<Core_clock_frequency>\d+)\sMHz::(?P<Video_memory>\d+)\sGB::(?P<Memory_bus_width>\d+)-bit$')
        match = pattern.match(string)
        if match:
            Stream_processor_count = int(match.group('Stream_processor_count'))
            Core_clock_frequency = int(match.group('Core_clock_frequency'))
            Video_memory = int(match.group('Video_memory'))
            Memory_bus_width = float(match.group('Memory_bus_width'))
        return [Stream_processor_count, Core_clock_frequency, Video_memory, Memory_bus_width]

    def get_dsp(self, string):
        # 3584::1480 MHz::11 GB::352-bit
        print(string)
        pattern = re.compile(
            r'^(?P<Stream_processor_count>\d+)::(?P<Core_clock_frequency>\d+)\sMHz::(?P<Video_memory>\d+)\sGB::(?P<Memory_bus_width>\d+)-bit$')
        match = pattern.match(string)
        if match:
            Stream_processor_count = int(match.group('Stream_processor_count'))
            Core_clock_frequency = int(match.group('Core_clock_frequency'))
            Video_memory = int(match.group('Video_memory'))
            Memory_bus_width = float(match.group('Memory_bus_width'))
        return [Stream_processor_count, Core_clock_frequency, Video_memory, Memory_bus_width]

    def get_tpu(self, string):
        # 3584::1480 MHz::11 GB::352-bit
        print(string)
        pattern = re.compile(
            r'^(?P<Stream_processor_count>\d+)::(?P<Core_clock_frequency>\d+)\sMHz::(?P<Video_memory>\d+)\sGB::(?P<Memory_bus_width>\d+)-bit$')
        match = pattern.match(string)
        if match:
            Stream_processor_count = int(match.group('Stream_processor_count'))
            Core_clock_frequency = int(match.group('Core_clock_frequency'))
            Video_memory = int(match.group('Video_memory'))
            Memory_bus_width = float(match.group('Memory_bus_width'))
        return [Stream_processor_count, Core_clock_frequency, Video_memory, Memory_bus_width]

    def get_device_item(self, file_names):
        from sklearn.preprocessing import LabelEncoder
        pattern = re.compile(r'^(?P<device_type>\w+)-(?P<unit>\w+)-(?P<devices>[\w-]+)-(?P<precision>\w+).pickle$')
        data = []
        for file_name in file_names:
            match = pattern.match(file_name)
            if match:
                data.append(match.groupdict())
        df = pandas.DataFrame(data).to_numpy()

        # 大语言模型进行数据增强
        try:
            makedir('./pretrained')
            with open(f'./pretrained/pretrained_{self.args.dataset}.pkl', 'rb') as f:
                device_info = pickle.load(f)
        except:
            print(df)
            device_info = []
            for i in range(len(df)):
                device_base_info = df[i, 2]
                device_more_info = self.chatgpt.get_device_more_info(device_base_info)
                if self.args.dataset == 'cpu':
                    device_info.append(self.get_cpu(device_more_info))
                elif self.args.dataset == 'gpu':
                    device_info.append(self.get_gpu(device_more_info))
            print(device_info)
            device_info = np.array(device_info)
            with open(f'./pretrained/pretrained_{self.args.dataset}.pkl', 'wb') as f:
                pickle.dump(device_info, f)
        # print(df)                                                  # 打印原始信息
        # print(device_info)                                         # 打印数据增强信息
        device_info = device_info / np.max(device_info, axis=0)  # 归一化
        # device_info = (device_info - np.mean(device_info, axis=0)) / np.std(device_info, axis=0)    # 归一化
        # print(device_info)
        df = np.delete(df, 2, axis=1)  # 删掉设备名
        # print(df)                                                # 打印删掉设备名的信息
        # 对每一列进行标签编码
        label_encoder = LabelEncoder()
        encoded_data = np.apply_along_axis(label_encoder.fit_transform, axis=0, arr=df)
        # print(encoded_data.shape, device_info.shape)
        final_data = np.concatenate([encoded_data, device_info], axis=1)
        # print(encoded_data)
        return final_data

    def get_device_item2(self, file_names):
        import re, pandas
        from sklearn.preprocessing import LabelEncoder
        pattern = re.compile(r'^(?P<device_type>\w+)-(?P<unit>\w+)-(?P<devices>[\w-]+)-(?P<precision>\w+).pickle$')
        data = []
        for file_name in file_names:
            match = pattern.match(file_name)
            if match:
                data.append(match.groupdict())
        df = pandas.DataFrame(data).to_numpy()
        # 对每一列进行标签编码
        label_encoder = LabelEncoder()
        encoded_data = np.apply_along_axis(label_encoder.fit_transform, axis=0, arr=df)
        final_data = np.concatenate([encoded_data], axis=1)
        return final_data

    # 只是读取大文件
    def load_data(self, args):
        import os
        file_names = os.listdir(args.path + '/' + args.dataset)
        pickle_files = [file for file in file_names if file.endswith('.pickle')]
        if args.llm:
            device_label = self.get_device_item(pickle_files)
            # print(device_label)
        else:
            device_label = self.get_device_item2(pickle_files)
        data = []
        for i in range(len(pickle_files)):
            pickle_file = args.path + pickle_files[i]
            with open(pickle_file, 'rb') as f:
                now = pickle.load(f)
            data.append([now])
        data = np.array(data)
        data = np.concatenate([device_label, data], axis=1)
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
            if args.dataset == 'cpu':
                idx = 8
            elif args.dataset == 'gpu':
                idx = 7
            for i in trange(len(data)):
                for key in (data[i][idx].keys()):
                    now = []
                    # 添加设备号
                    for j in range(len(data[0]) - 1):
                        now.append(data[i][j])
                    # 获得编码取嵌入
                    op_idx = self.get_idx(key)
                    for j in range(len(op_idx)):
                        now.append(op_idx[j])
                    # now.append(4)  # 这个是最后一个输出节点
                    # 图和 one-hot Feature
                    matrix, features = self.get_graph(key)
                    now.append(matrix)
                    now.append(features)
                    y = data[i][idx][key]
                    now.append(y)
                    tensor.append(now)
            tensor = np.array(tensor)
        return tensor

    def get_pytorch_index(self, data):
        return data

    def get_idx(self, op_seq):
        # 全部代码
        op_seq = list(op_seq)
        matrix, label = self.get_matrix_and_ops(op_seq)
        matrix, features = self.get_adjacency_and_features(matrix, label)

        def get_op_idx(features):
            result = [row.index(1) if 1 in row else 5 for row in features]
            return np.array(result)
        op_idx = get_op_idx(features)
        """
            本人定义：0 con1 1 con3 2 max3 3 input 4 output 5 None
            数据集定义：
                0 : None 5
                1 : None 5
                2 ： 0  con1
                3 ： 1  con2
                4 ： 2  max3
                input : 3
                output : 4
                [0, 1, 2, 3, 4, 5]
                [5, 5, 0, 1, 2, 3]
        """
        # print(op_seq)
        # print(label)
        # print(op_idx)
        # print('-' * 100)
        return op_idx

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


def get_train_valid_test_dataset(tensor, args):
    p = np.random.permutation(len(tensor))
    tensor = tensor[p]

    X = tensor[:, :-1]
    Y = tensor[:, -1].reshape(-1, 1)
    # max_value = Y.max()
    max_value = 1
    Y /= max_value

    # train_size = int(len(tensor) * args.density)  # Assuming 900 samples for training
    train_size = int(len(tensor) * args.density)  # Assuming 900 samples for training
    valid_size = int(100)  # Assuming 113 samples for validation
    if args.dataset == 'gpu':
        valid_size = int(100 * 2)

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
        args.log.only_print(f'Train_length : {len(self.train_loader.dataset)} Valid_length : {len(self.valid_loader.dataset)} Test_length : {len(self.test_loader.dataset)}')

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
        device_features = self.indices[idx, :-3]
        device_features = list(device_features)
        matrix, features = self.indices[idx, -3], self.indices[idx, -2]
        matrix, features = list(matrix), list(features)
        value = torch.as_tensor(self.indices[idx, -1])  # 最后一列作为真实值
        return device_features, matrix, features, value

    def __len__(self):
        return self.indices.shape[0]

    def delete_zero_row(self, tensor):
        row_sums = tensor.sum(axis=1)
        nonzero_rows = (row_sums != 0).nonzero().squeeze()
        filtered_tensor = tensor[nonzero_rows]
        return filtered_tensor


class GraphConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, weight_init='thomas', bias_init='thomas'):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.torch.nn.Parameter(torch.DoubleTensor(in_features, out_features))
        if bias:
            self.bias = torch.torch.nn.Parameter(torch.DoubleTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.weight_init = weight_init
        self.bias_init = bias_init
        self.reset_parameters()

    def reset_parameters(self):
        self.init_tensor(self.weight, self.weight_init, 'relu')
        self.init_tensor(self.bias, self.bias_init, 'relu')


    def forward(self, adjacency, features):
        support = torch.matmul(features, self.weight)
        output = torch.bmm(adjacency, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

    def init_tensor(self, tensor, init_type, nonlinearity):
        if tensor is None or init_type is None:
            return
        if init_type == 'thomas':
            size = tensor.size(-1)
            stdv = 1. / math.sqrt(size)
            torch.nn.init.uniform_(tensor, -stdv, stdv)
        elif init_type == 'kaiming_normal_in':
            torch.nn.init.kaiming_normal_(tensor, mode='fan_in', nonlinearity=nonlinearity)
        elif init_type == 'kaiming_normal_out':
            torch.nn.init.kaiming_normal_(tensor, mode='fan_out', nonlinearity=nonlinearity)
        elif init_type == 'kaiming_uniform_in':
            torch.nn.init.kaiming_uniform_(tensor, mode='fan_in', nonlinearity=nonlinearity)
        elif init_type == 'kaiming_uniform_out':
            torch.nn.init.kaiming_uniform_(tensor, mode='fan_out', nonlinearity=nonlinearity)
        elif init_type == 'orthogonal':
            torch.nn.init.orthogonal_(tensor, gain=torch.nn.init.calculate_gain(nonlinearity))
        else:
            raise ValueError(f'Unknown initialization type: {init_type}')


class GCN(torch.nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.args = args

        num_features = 6
        num_layers = 4
        num_hidden = 600
        dropout_ratio = 0.02
        weight_init = 'thomas'
        bias_init = 'thomas'
        binary_classifier = False
        augments = 0

        self.nfeat = num_features
        self.nlayer = num_layers
        self.nhid = num_hidden
        self.dropout_ratio = dropout_ratio

        self.gc = torch.nn.ModuleList([GraphConvolution(self.nfeat if i == 0 else self.nhid, self.nhid, bias=True,
                                                        weight_init=weight_init, bias_init=bias_init) for i in
                                       range(self.nlayer)])
        self.bn = torch.nn.ModuleList([torch.nn.LayerNorm(self.nhid).double() for i in range(self.nlayer)])
        self.relu = torch.nn.ModuleList([torch.nn.ReLU().double() for i in range(self.nlayer)])
        self.dropout = torch.nn.ModuleList([torch.nn.Dropout(self.dropout_ratio).double() for i in range(self.nlayer)])

        if not binary_classifier:
            self.fc = torch.nn.Linear(self.nhid + augments, 1).double()
        else:
            if binary_classifier == 'naive':
                self.fc = torch.nn.Linear(self.nhid + augments, 1).double()
            elif binary_classifier == 'oneway' or binary_classifier == 'oneway-hard':
                self.fc = torch.nn.Linear((self.nhid + augments) * 2, 1).double()
            else:
                self.fc = torch.nn.Linear((self.nhid + augments) * 2, 2).double()

            if binary_classifier != 'oneway' and binary_classifier != 'oneway-hard':
                self.final_act = torch.nn.LogSoftmax(dim=1)
            else:
                self.final_act = torch.nn.Sigmoid()
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

        a = []
        b = []
        for i in range(len(adjacency)):
            a.append(adjacency[i])
            b.append(features[i])

        adjacency = torch.DoubleTensor(a)
        features = torch.DoubleTensor(b)

        augments = None
        if not self.binary_classifier:
            x = self.forward_single_model(adjacency, features)
            x = x[:, 0]  # use global node
            if augments is not None:
                x = torch.cat([x, augments], dim=1)
            return x
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


class Device_encoder(torch.nn.Module):
    def __init__(self, args):
        super(Device_encoder, self).__init__()
        self.args = args
        self.dim = args.dimension

        self.platform_embeds = torch.nn.Embedding(6, self.dim)
        self.device_embeds = torch.nn.Embedding(6, self.dim)
        self.device_name_embeds = torch.nn.Embedding(6, self.dim)
        self.precision_embeds = torch.nn.Embedding(6, self.dim)

        self.mlp = torch.nn.Linear(6 * 9, self.dim)

        # 第二个想法
        if args.llm:
            input_dim = None
            if args.dataset == 'cpu':
                input_dim = 5
            elif args.dataset == 'gpu':
                input_dim = 4
            self.transfer = torch.nn.Linear(input_dim, self.dim)
        else:
            self.transfer = torch.nn.Linear(1, self.dim)

        self.op_embeds = torch.nn.Embedding(6, self.dim)

        self.initialize()
        self.cache = {}

    def initialize(self):
        pass

    # no embeds
    def forward(self, inputs):
        if self.args.llm:
            platformIdx, deviceIdx, device_info_llm, precisionIdx, op_idx = self.get_inputs(inputs)
        else:
            platformIdx, deviceIdx, device_name_Idx, precisionIdx, op_idx = self.get_inputs2(inputs)

        one_hot_encoded = torch.nn.functional.one_hot(op_idx.long(), num_classes=6).to(torch.float64).reshape(len(op_idx), -1)
        seq_embeds = self.mlp(one_hot_encoded)
        device_name_embeds = self.transfer(device_info_llm.double())

        # Final interaction
        final_input = torch.cat([
            seq_embeds, device_name_embeds
        ], dim=-1)
        return final_input


    def get_inputs(self, inputs):
        if self.args.dataset == 'cpu':
            platformIdx, deviceIdx, precisionIdx, \
                frequency, cores, threads, memory_size, memory_speed, \
                firstIdx, secondIdx, thirdIdx, fourthIdx, fifthIdx, sixthIdx, seventhIdx, eighthIdx, nineIdx = inputs
            device_info_llm = torch.vstack([frequency, cores, threads, memory_size, memory_speed]).T
        elif self.args.dataset == 'gpu':
            platformIdx, deviceIdx, precisionIdx, \
                Stream_processor_count, Core_clock_frequency, Video_memory, Memory_bus_width, \
                firstIdx, secondIdx, thirdIdx, fourthIdx, fifthIdx, sixthIdx, seventhIdx, eighthIdx, nineIdx = inputs
            device_info_llm = torch.vstack([Stream_processor_count, Core_clock_frequency, Video_memory, Memory_bus_width]).T

        # firstIdx, secondIdx, thirdIdx, fourthIdx,\
        op_idx = torch.vstack([firstIdx, secondIdx, thirdIdx, fourthIdx, fifthIdx, sixthIdx, seventhIdx, eighthIdx, nineIdx]).T
        op_idx = op_idx.to(torch.long)
        return platformIdx.long(), deviceIdx.long(), device_info_llm.float(), precisionIdx.long(), op_idx.long(),


    def get_inputs2(self, inputs):
        firstIdx, secondIdx, thirdIdx, fourthIdx, \
            fifthIdx, sixthIdx, seventhIdx, eighthIdx, ninthIdx, tenthIdx, elemIdx = inputs
        op_idx = torch.vstack([fifthIdx, sixthIdx, seventhIdx, eighthIdx, ninthIdx, tenthIdx, elemIdx])
        op_idx = op_idx.to(torch.long)
        # 获得计算节点信息
        platformIdx = firstIdx
        deviceIdx = secondIdx
        device_name_Idx = thirdIdx
        precisionIdx = fourthIdx
        return platformIdx.long(), deviceIdx.long(), device_name_Idx.long(), precisionIdx.long(), op_idx.long()


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim // 2),  # FFN
            torch.nn.LayerNorm(hidden_dim // 2),  # LayerNorm
            torch.nn.ReLU(),  # ReLU
            torch.nn.Linear(hidden_dim // 2, hidden_dim // 2),  # FFN
            torch.nn.LayerNorm(hidden_dim // 2),  # LayerNorm
            torch.nn.ReLU(),  # ReLU
            torch.nn.Linear(hidden_dim // 2, output_dim)  # y
        )

    def forward(self, x):
        return self.layer(x)

class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.gcn = GCN(args)
        self.device_encoder = Device_encoder(args)
        final_input = args.dimension * 2 + 600
        self.mlp = MLP(final_input, final_input // 2, 1)

    def forward(self, device_features, matrix, features):
        device_features = self.device_encoder(device_features)
        graph_features = self.gcn(matrix, features)
        y = self.mlp(torch.cat([device_features, graph_features], dim = -1))
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
            device_features, matrix, features, value = train_Batch
            pred = self.forward(device_features, matrix, features)
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
            device_features, matrix, features, value = valid_Batch
            pred = self.forward(device_features, matrix, features)
            val_loss += self.loss_function(pred, value).item()
            preds[writeIdx:writeIdx + len(pred)] = pred
            reals[writeIdx:writeIdx + len(value)] = value
            writeIdx += len(pred)
        if self.epochs > 20:
            self.scheduler.step(val_loss)
        valid_error = ErrorMetrics(reals * dataModule.max_value, preds * dataModule.max_value)
        return valid_error

    def test_one_epoch(self, dataModule):
        writeIdx = 0
        preds = torch.zeros((len(dataModule.test_loader.dataset),)).to(self.args.device)
        reals = torch.zeros((len(dataModule.test_loader.dataset),)).to(self.args.device)
        for test_Batch in dataModule.test_loader:
            device_features, matrix, features, value = test_Batch
            pred = self.forward(device_features, matrix, features)
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
    import dgl
    if len(batch[0]) == 2:  # 当 self.args.exper != 6
        return default_collate(batch)
    elif len(batch[0]) == 3:
        op_idxs, graphs, values = zip(*batch)
        # graphs = dgl.batch(graphs)
        graphs = list(graphs)
        op_idxs = list(op_idxs)
        values = default_collate(values)
        return op_idxs, graphs, values
    else:
        device_features, matrix, features, value = zip(*batch)
        device_features = default_collate(device_features)
        matrix = list(matrix)
        features = list(features)
        value = default_collate(value)
        return device_features, matrix, features, value


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
    parser.add_argument('--model', type=str, default='gcn_llm')  #

    # Experiment
    parser.add_argument('--density', type=float, default=0.10)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--record', type=int, default=1)
    parser.add_argument('--program_test', type=int, default=0)
    parser.add_argument('--experiment', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--path', nargs='?', default='./datasets/')

    # Training tool
    parser.add_argument('--device', type=str, default='cpu')  # gpu cpu mps
    parser.add_argument('--bs', type=int, default=1)  #
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--decay', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=30)
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



