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


    def preprocess_data(self, data, args):
        try:
            tensor = pickle.load(open(f'./pretrained/tensor_{torch.initial_seed()}.pkl', 'rb'))
        except:
            tensor = []
            for i in range(len(data)):
                for key in (data[i][0].keys()):
                    now = []
                    now.append(i)
                    # 添加设备号
                    # print(key)
                    for item in key:
                        now.append(item)
                    y = data[i][0][key]
                    now.append(y)
                    tensor.append(now)
                    # print(now)
            tensor = np.array(tensor)
        return tensor

    def get_pytorch_index(self, data):
        return data


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
        inputs = self.indices[idx, :-1]
        value = self.indices[idx, -1]
        return inputs, value

    def __len__(self):
        return self.indices.shape[0]


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(input_dim + 1, hidden_dim // 2),  # FFN
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
        self.input_dim = 6
        self.hidden_dim = args.dimension
        self.mlp = MLP(self.input_dim, self.hidden_dim, 1)

    def forward(self, inputs):
        y = self.mlp(inputs)
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
            inputs, value = train_Batch
            pred = self.forward(inputs)
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
            inputs, value = valid_Batch
            pred = self.forward(inputs)
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
            inputs, value = test_Batch
            pred = self.forward(inputs)
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
        batch_size=args.bs * 16,
        drop_last=False,
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.bs * 16,  # 8192
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
    parser.add_argument('--model', type=str, default='MLP')  #

    # Experiment
    parser.add_argument('--density', type=float, default=0.05)
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
    parser.add_argument('--patience', type=int, default=50)
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



