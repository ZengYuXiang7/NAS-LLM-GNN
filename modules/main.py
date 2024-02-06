# coding : utf-8
# Author : yuxiang Zeng
import os
import time
import torch
import numpy as np
import collections
from modules.datasets import get_exper
from modules.models import get_model
from utils.datamodule import DataModule
from utils.logger import Logger
from utils.monitor import EarlyStopping
from utils.utils import set_seed, set_settings

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
global log
torch.set_default_dtype(torch.double)


# Run Experiments
def RunOnce(args, runId, Runtime, log):
    # Set seed
    set_seed(args.seed + runId)

    # Initialize
    exper = get_exper(args)
    datamodule = DataModule(exper, args)
    model = get_model(args)
    monitor = EarlyStopping(args.patience)

    # Setup training tool
    model.setup_optimizer(args)
    model.max_value = datamodule.max_value

    train_time = []
    valid_error = None
    for epoch in range(args.epochs):
        model.set_epochs(epoch)
        epoch_loss, time_cost = model.train_one_epoch(datamodule)
        valid_error = model.valid_one_epoch(datamodule)
        monitor.track(epoch, model.state_dict(), valid_error['MAE'])
        train_time.append(time_cost)
        if args.verbose and epoch % args.verbose == 0:
            log.only_print(
                f"Round={runId + 1} Epoch={epoch + 1:02d} Loss={epoch_loss:.4f} vMAE={valid_error['MAE']:.4f} vRMSE={valid_error['RMSE']:.4f} vNMAE={valid_error['NMAE']:.4f} vNRMSE={valid_error['NRMSE']:.4f} time={sum(train_time):.1f} s")
            log.only_print(f"Acc = [1%={valid_error['Acc'][0]:.4f}, 5%={valid_error['Acc'][1]:.4f}, 10%={valid_error['Acc'][2]:.4f}]")
        if monitor.early_stop:
            break

    model.load_state_dict(monitor.best_model)
    sum_time = sum(train_time[: monitor.best_epoch])
    results = model.test_one_epoch(datamodule) if args.valid else valid_error
    log.only_print(f'Round={runId + 1} BestEpoch={monitor.best_epoch:d} vMAE={valid_error["MAE"]:.4f} vRMSE={valid_error["RMSE"]:.4f} vNMAE={valid_error["NMAE"]:.4f} vNRMSE={valid_error["NRMSE"]:.4f} Training_time={sum_time:.1f} s\n')
    log.only_print(f"Acc = [1%={valid_error['Acc'][0]:.4f}, 5%={valid_error['Acc'][1]:.4f}, 10%={valid_error['Acc'][2]:.4f}]")
    log(f'Round={runId + 1} BestEpoch={monitor.best_epoch:d} tMAE={results["MAE"]:.4f} tRMSE={results["RMSE"]:.4f} tNMAE={results["NMAE"]:.4f} tNRMSE={results["NRMSE"]:.4f} Training_time={sum_time:.1f} s\n')
    log(f"Acc = [1%={results['Acc'][0]:.4f}, 5%={results['Acc'][1]:.4f}, 10%={results['Acc'][2]:.4f}]")

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


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rounds', type=int, default=5)

    parser.add_argument('--dataset', type=str, default='1')  #
    parser.add_argument('--exper', type=int, default=4)  #
    parser.add_argument('--model', type=str, default='4')  # NeuTF, 2, 3

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
    parser.add_argument('--bs', type=int, default=64)  #
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr_step', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--saved', type=int, default=1)

    parser.add_argument('--loss_func', type=str, default='L1Loss')
    parser.add_argument('--optim', type=str, default='AdamW')

    # Hyper parameters
    parser.add_argument('--dimension', type=int, default=64)
    parser.add_argument('--windows', type=int, default=5)

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


