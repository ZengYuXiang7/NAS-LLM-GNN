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
from modules.models import get_model
from utils.dataloader import get_dataloaders
from utils.datamodule import DataModule
from utils.logger import Logger
from utils.metamodel import MetaModel
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

    def load_data(self, args):
        import os
        file_names = os.listdir(args.path + '/' + args.dataset)
        # print(file_names)
        pickle_files = [file for file in file_names if file.endswith('.pickle')]
        data = []
        for i in range(len(pickle_files)):
            pickle_file = args.path + pickle_files[i]
            # print(pickle_file)
            with open(pickle_file, 'rb') as f:
                now = pickle.load(f)
            data.append(now)
        return data

    def preprocess_data(self, data, args):
        tensor = []
        for i in range(len(data)):
            for key in data[i].keys():
                now = []
                now.append(i)
                for x in key:
                    now.append(x)
                y = data[i][key]
                now.append(y)
                tensor.append(now)
        tensor = np.array(tensor)
        return tensor

    def get_pytorch_index(self, data):
        return torch.as_tensor(data)



# 模型三
class NAS_Model_3(MetaModel):
    def __init__(self, args):
        super(NAS_Model_3, self).__init__(args)
        self.args = args
        self.dim = args.dimension

        self.first_embeds = torch.nn.Embedding(15, self.dim)
        self.second_embeds = torch.nn.Embedding(5, self.dim)
        self.third_embeds = torch.nn.Embedding(5, self.dim)
        self.fourth_embeds = torch.nn.Embedding(5, self.dim)
        self.fifth_embeds = torch.nn.Embedding(5, self.dim)
        self.sixth_embeds = torch.nn.Embedding(5, self.dim)
        self.seventh_embeds = torch.nn.Embedding(5, self.dim)

        input_dim = 7 * self.dim
        self.NeuCF = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim // 2),  # FFN
            torch.nn.LayerNorm(input_dim // 2),  # LayerNorm
            torch.nn.ReLU(),  # ReLU
            torch.nn.Linear(input_dim // 2, input_dim // 2),  # FFN
            torch.nn.LayerNorm(input_dim // 2),  # LayerNorm
            torch.nn.ReLU(),  # ReLU
            torch.nn.Linear(input_dim // 2, 1)  # y
        )
        self.cache = {}
        self.initialize()

    def initialize(self):
        # torch.nn.init.kaiming_normal_(self.first_embeds.weight)
        torch.nn.init.kaiming_normal_(self.second_embeds.weight)
        torch.nn.init.kaiming_normal_(self.third_embeds.weight)
        torch.nn.init.kaiming_normal_(self.fourth_embeds.weight)
        torch.nn.init.kaiming_normal_(self.fifth_embeds.weight)
        torch.nn.init.kaiming_normal_(self.sixth_embeds.weight)
        torch.nn.init.kaiming_normal_(self.seventh_embeds.weight)

    def forward(self, inputs, train = True):
        firstIdx, secondIdx, thirdIdx, fourthIdx, fifthIdx, sixthIdx, seventhIdx = self.get_inputs(inputs)
        first_embeds = self.first_embeds(firstIdx)

        second_embeds = self.second_embeds(secondIdx)
        third_embeds = self.third_embeds(thirdIdx)
        fourth_embeds = self.fourth_embeds(fourthIdx)
        fifth_embeds = self.fifth_embeds(fifthIdx)
        sixth_embeds = self.sixth_embeds(sixthIdx)
        seventh_embeds = self.seventh_embeds(seventhIdx)

        estimated = self.NeuCF(
            torch.cat(
                (first_embeds, second_embeds, third_embeds, fourth_embeds, fifth_embeds, sixth_embeds, seventh_embeds),
                dim=-1)).sigmoid().reshape(-1)
        return estimated

    def prepare_test_model(self):
        pass

    def get_inputs(self, inputs):
        firstIdx, secondIdx, thirdIdx, fourthIdx, fifthIdx, sixthIdx, seventhIdx = inputs
        return firstIdx.long(), secondIdx.long(), thirdIdx.long(), fourthIdx.long(), fifthIdx.long(), sixthIdx.long(), seventhIdx.long()


def RunOnce(args, runId, Runtime, log):
    # Set seed
    set_seed(args.seed + runId)

    # Initialize
    args.exper = 3
    args.model = '3'
    exper = experiment(args)
    datamodule = DataModule(exper, args)
    model = get_model(args)
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

    parser.add_argument('--dataset', type=str, default='cpu')  #
    parser.add_argument('--model', type=str, default='embed')  #

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
    parser.add_argument('--patience', type=int, default=100)
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



