# coding : utf-8
# Author : yuxiang Zeng

from modules.datasets.train_test_split import get_train_valid_test_dataset
from utils.dataloader import get_dataloaders

import torch
from torch.utils.data import Dataset


# 数据集定义
class DataModule:
    def __init__(self, exper_type, args):
        self.args = args
        self.path = args.path

        # 加载原始数据
        self.data = exper_type.load_data(args)

        # 预处理数据
        self.data = exper_type.preprocess_data(self.data, args)

        # 切分训练测试
        self.train_tensor, self.valid_tensor, self.test_tensor, self.max_value = get_train_valid_test_dataset(self.data, args)

        # 装载数据
        self.train_set, self.valid_set, self.test_set = self.get_dataset(self.train_tensor, self.valid_tensor, self.test_tensor, exper_type, args)

        # 装进pytorch
        self.train_loader, self.valid_loader, self.test_loader = get_dataloaders(self.train_set, self.valid_set, self.test_set, args)

        # 基本信息
        args.log.only_print(f'Train_length : {len(self.train_loader.dataset)} Valid_length : {len(self.valid_loader.dataset)} Test_length : {len(self.test_loader.dataset)}')

    def get_tensor(self):
        return self.train_tensor, self.valid_tensor, self.test_tensor

    def trainLoader(self):
        return self.train_loader

    def validLoader(self):
        return self.valid_loader

    def testLoader(self):
        return self.test_loader

    def fullLoader(self):
        return self.fullloader

    def get_dataset(self, train_tensor, valid_tensor, test_tensor, exper_type, args):
        return TensorDataset(train_tensor, exper_type, args), TensorDataset(valid_tensor, exper_type, args), TensorDataset(test_tensor, exper_type, args)

class TensorDataset(torch.utils.data.Dataset):

    def __init__(self, tensor, exper_type, args):
        self.args = args
        self.tensor = tensor
        self.indices = exper_type.get_pytorch_index(tensor)
        self.indices = self.delete_zero_row(self.indices)

    def __getitem__(self, idx):
        if self.args.exper not in [6, 7]:
            output = self.indices[idx, :-1]  # 去掉最后一列
            inputs = tuple(torch.as_tensor(output[i]) for i in range(output.shape[0]))
            value = torch.as_tensor(self.indices[idx, -1])  # 最后一列作为真实值
            return inputs, value
        else:
            op, graph = self.indices  # 去掉最后一列
            op_idx = torch.as_tensor(op[idx, :-1])
            value = torch.as_tensor(op[idx, -1])  # 最后一列作为真实值
            graph = graph[idx]
            # inputs = op_idx, gr
            return op_idx, graph, value


    def __len__(self):
        if self.args.exper not in [6, 7]:
            return self.indices.shape[0]
        else:
            return self.indices[0].shape[0]


    def delete_zero_row(self, tensor):
        if self.args.exper not in [6, 7]:
            row_sums = tensor.sum(axis=1)
            nonzero_rows = (row_sums != 0).nonzero().squeeze()
            filtered_tensor = tensor[nonzero_rows]
        else:
            idx, graph = tensor
            row_sums = idx.sum(axis=1)
            nonzero_rows = (row_sums != 0).nonzero().squeeze()
            idx_tensor = idx[nonzero_rows]
            graph_tensor = graph[nonzero_rows]
            filtered_tensor = idx_tensor, graph_tensor
        return filtered_tensor
