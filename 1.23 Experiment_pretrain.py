# coding : utf-8
# Author : yuxiang Zeng
import os
import time
import numpy as np
import collections

import torch

from modules.datasets import get_exper
from modules.models import get_model
from utils.datamodule import DataModule
from utils.logger import Logger
from utils.monitor import EarlyStopping
from utils.utils import set_seed, set_settings

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

global log


# Run Experiments
def RunOnce(args, runId, Runtime, log):
    args.path = './datasets/' + 'cpu' + '/'
    args.dataset_type = 'cpu'

    # Set seed
    set_seed(args.seed + runId)

    # Pretrain experiment
    args.exper = 8
    args.model = str(8)
    exper = get_exper(args)
    datamodule = DataModule(exper, args)
    model = get_model(args)
    monitor = EarlyStopping(args.patience)
    model.setup_optimizer(args)
    model.max_value = datamodule.max_value

    train_time = []
    valid_error = None
    for epoch in range(args.epochs):
        epoch_loss, time_cost = model.train_one_epoch(datamodule)
        valid_error = model.valid_one_epoch(datamodule)
        monitor.track(epoch, model.state_dict(), valid_error['MAE'])
        train_time.append(time_cost)
        if args.verbose and epoch % args.verbose == 0:
            log.only_print(
                f"Round={runId + 1} Epoch={epoch + 1:02d} Loss={epoch_loss:.4f} vMAE={valid_error['MAE']:.4f} vRMSE={valid_error['RMSE']:.4f} vNMAE={valid_error['NMAE']:.4f} vNRMSE={valid_error['NRMSE']:.4f} time={sum(train_time):.1f} s")
        if monitor.early_stop:
            break

    model.load_state_dict(monitor.best_model)
    sum_time = sum(train_time[: monitor.best_epoch])
    results = model.test_one_epoch(datamodule) if args.valid else valid_error
    log(f'Round={runId + 1} BestEpoch={monitor.best_epoch:d} MAE={results["MAE"]:.4f} RMSE={results["RMSE"]:.4f} NMAE={results["NMAE"]:.4f} NRMSE={results["NRMSE"]:.4f} Training_time={sum_time:.1f} s\n')

    # Get pretrained embeddings
    first_embeds = model.first_embeds.weight.detach()
    second_embeds = model.second_embeds.weight.detach()
    third_embeds = model.third_embeds.weight.detach()
    fourth_embeds = model.fourth_embeds.weight.detach()
    fifth_embeds = model.fifth_embeds.weight.detach()
    sixth_embeds = model.sixth_embeds.weight.detach()
    seventh_embeds = model.seventh_embeds.weight.detach()
    eighth_embeds = model.eighth_embeds.weight.detach()

    features = np.stack([first_embeds, second_embeds, third_embeds, fourth_embeds,
                         fifth_embeds, sixth_embeds, seventh_embeds, eighth_embeds])
    features = np.mean(features, axis=1)

    # Formal model
    args.exper = 7
    args.model = str(7)
    exper = get_exper(args)
    datamodule = DataModule(exper, args)
    model = get_model(args)
    monitor = EarlyStopping(args.patience)

    # Initial Embedding
    model.op_embeds = torch.as_tensor(features)
    model.setup_optimizer(args)
    model.max_value = datamodule.max_value

    train_time = []
    valid_error = None
    for epoch in range(args.epochs):
        epoch_loss, time_cost = model.train_one_epoch(datamodule)
        valid_error = model.valid_one_epoch(datamodule)
        monitor.track(epoch, model.state_dict(), valid_error['MAE'])
        train_time.append(time_cost)
        if args.verbose and epoch % args.verbose == 0:
            log.only_print(
                f"Round={runId + 1} Epoch={epoch + 1:02d} Loss={epoch_loss:.4f} vMAE={valid_error['MAE']:.4f} vRMSE={valid_error['RMSE']:.4f} vNMAE={valid_error['NMAE']:.4f} vNRMSE={valid_error['NRMSE']:.4f} time={sum(train_time):.1f} s")
        if monitor.early_stop:
            break

    model.load_state_dict(monitor.best_model)
    sum_time = sum(train_time[: monitor.best_epoch])
    results = model.test_one_epoch(datamodule) if args.valid else valid_error
    log(f'Round={runId + 1} BestEpoch={monitor.best_epoch:d} MAE={results["MAE"]:.4f} RMSE={results["RMSE"]:.4f} NMAE={results["NMAE"]:.4f} NRMSE={results["NRMSE"]:.4f} Training_time={sum_time:.1f} s\n')

    return {
        'MAE': results["MAE"],
        'RMSE': results["RMSE"],
        'NMAE': results["NMAE"],
        'NRMSE': results["NRMSE"],
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

    log('*' * 20 + 'Experiment Success' + '*' * 20 + '\n')

    return metrics


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rounds', type=int, default=5)

    parser.add_argument('--dataset', type=str, default='1')  #
    parser.add_argument('--exper', type=int, default=7)  #
    parser.add_argument('--model', type=str, default='7')  # NeuTF, 2, 3

    # Experiment
    parser.add_argument('--density', type=float, default=0.10)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--record', type=int, default=0)
    parser.add_argument('--program_test', type=int, default=0)
    parser.add_argument('--valid', type=int, default=1)
    parser.add_argument('--experiment', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=10)  # 10
    parser.add_argument('--path', nargs='?', default='./datasets/')

    # Training tool
    parser.add_argument('--device', type=str, default='cpu')  # gpu cpu mps
    parser.add_argument('--bs', type=int, default=32)  #
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--decay', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr_step', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--saved', type=int, default=1)

    parser.add_argument('--loss_func', type=str, default='L1Loss')
    parser.add_argument('--optim', type=str, default='AdamW')

    # Hyper parameters
    parser.add_argument('--dimension', type=int, default=64)

    # Other Experiment
    parser.add_argument('--ablation', type=int, default=0)
    args = parser.parse_args([])
    return args


if __name__ == '__main__':
    # Setup Arguments
    args = get_args()
    set_settings(args)

    # Setup Logger
    log = Logger(args)

    # Record Experiments Config
    log(str(args))
    args.log = log

    # Run Experiments
    RunExperiments(log, args)
