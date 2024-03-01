# coding : utf-8
# Author : yuxiang Zeng
import collections
import math
import time

import pickle
import numpy as np
import argparse
import copy

from skimage.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, ParameterGrid
from torch.utils.data import DataLoader
from tqdm import *
import torch
from utils.dataloader import get_dataloaders
from utils.logger import Logger
from utils.metrics import ErrorMetrics
from utils.monitor import EarlyStopping
from utils.trainer import get_loss_function, get_optimizer
from utils.utils import optimizer_zero_grad, optimizer_step, lr_scheduler_step, set_settings, set_seed

global log
torch.set_default_dtype(torch.double)

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


    def preprocess_data(self, data, args):
        try:
            tensor = pickle.load(open(f'./pretrained/tensor_{torch.initial_seed()}.pkl', 'rb'))
        except:
            tensor = []
            for i in range(len(data)):
                for key in (data[i][0].keys()):
                    now = []
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


def get_train_valid_test_dataset(tensor, args):
    np.random.shuffle(tensor)

    X = tensor[:, :-1]
    Y = tensor[:, -1].reshape(-1, 1)
    # max_value = Y.max()
    max_value = 1
    Y /= max_value

    train_size = int(900)  # Assuming 900 samples for training
    valid_size = int(100)  # Assuming 113 samples for validation

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
        # args.log.only_print(f'Train_length : {len(self.train_tensor)} Valid_length : {len(self.valid_tensor)} Test_length : {len(self.test_tensor)}')




class Model(torch.torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args

    def forward(self, adjacency, features):
        pass

    def machine_learning_model_train_evaluation(self, train_x, train_y, valid_x, valid_y, test_x, test_y, max_value):
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import ParameterGrid
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.svm import SVR
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor
        # print(train_x.shape, train_y.shape, valid_x.shape, valid_y.shape, test_x.shape, test_y.shape)
        models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'KNeighborsRegressor': KNeighborsRegressor(),
            'SVR': SVR(),
            'DecisionTreeRegressor': DecisionTreeRegressor(),
            'RandomForestRegressor': RandomForestRegressor(),
            'GradientBoostingRegressor': GradientBoostingRegressor(),
        }
        param_grids = {
            'Ridge': {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]},
            'Lasso': {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]},
            'KNeighborsRegressor': {'n_neighbors': [3, 5, 7, 9, 11, 15]},
            'SVR': {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'kernel': ['rbf', 'linear']},
            'DecisionTreeRegressor': {
                'max_depth': [None, 5, 10, 15, 20],
                'min_samples_split': [2, 5, 10]
            },
            'RandomForestRegressor': {
                'n_estimators': [10, 50, 100],
                'max_depth': [None, 5, 10, 15, 20],
                'min_samples_split': [2, 5, 10]
            },
            'GradientBoostingRegressor': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 1],
                'max_depth': [3, 5, 7]
            }
        }
        # 在开始训练模型之前，初始化一个空字典来存储结果
        results_dict = {}
        for name, model in models.items():
            # print('-' * 80)
            # print(f"正在训练模型: {name}")
            if name in param_grids:
                best_score = float('inf')
                best_params = None
                for params in ParameterGrid(param_grids[name]):
                    model.set_params(**params)
                    model.fit(train_x, train_y)
                    predictions = model.predict(valid_x)
                    score = mean_squared_error(valid_y, predictions)
                    if score < best_score:
                        best_score = score
                        best_params = params
                # print(f"{name} 最佳参数: {best_params}")
                model.set_params(**best_params)
                model.fit(train_x, train_y)
            else:
                model.fit(train_x, train_y)
            predict_test_y = model.predict(test_x)
            results_test = ErrorMetrics(predict_test_y * max_value, test_y * max_value)
            # print(f"测试集上的表现 - MAE={results_test['MAE']:.4f}, RMSE={results_test['RMSE']:.4f}, NMAE={results_test['NMAE']:.4f}, NRMSE={results_test['NRMSE']:.4f}")
            # print(f"Acc = [1%={results_test['Acc'][0]:.4f}, 5%={results_test['Acc'][1]:.4f}, 10%={results_test['Acc'][2]:.4f}]")
            results_dict[name] = results_test
        return results_dict
def RunOnce(args, runId, Runtime, log):
    # Set seed
    set_seed(args.seed + runId)

    # Initialize
    exper = experiment(args)
    datamodule = DataModule(exper, args)
    model = Model(args)

    # Prepare the data for machine learning
    train_x, train_y = datamodule.train_tensor[:, :-1], datamodule.train_tensor[:, -1]
    valid_x, valid_y = datamodule.valid_tensor[:, :-1], datamodule.valid_tensor[:, -1]
    test_x, test_y = datamodule.test_tensor[:, :-1], datamodule.test_tensor[:, -1]
    max_value = datamodule.max_value
    results = model.machine_learning_model_train_evaluation(train_x, train_y, valid_x, valid_y, test_x, test_y, max_value)
    return results


def RunExperiments(log, args):
    log('*' * 20 + 'Experiment Start' + '*' * 20)
    metrics = collections.defaultdict(list)

    for runId in trange(args.rounds, desc=f'机器学习大实验'):
        runHash = int(time.time())
        results = RunOnce(args, runId, runHash, log)
        for model_name, model_results in results.items():
            for metric_name, metric_value in model_results.items():
                if metric_name == 'Acc':
                    continue
                    # for acc_name, acc_value in zip(['1', '5', '10'], metric_value):
                    #     metrics[f"{model_name}_{metric_name}_{acc_name}"].append(acc_value)
                else:
                    metrics[f"{model_name}_{metric_name}"].append(metric_value)
    log('*' * 20 + 'Experiment Results:' + '*' * 20)
    for key in metrics:
        log(f'{key}: {np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f}')
    if args.record:
        log.save_result(metrics)
    log('*' * 20 + 'Experiment Success' + '*' * 20 + '\n')

    return metrics


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rounds', type=int, default=5)

    parser.add_argument('--dataset', type=str, default='rt')  #
    parser.add_argument('--model', type=str, default='CF')  #

    # Experiment
    parser.add_argument('--density', type=float, default=0.01)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--record', type=int, default=1)
    parser.add_argument('--program_test', type=int, default=0)
    parser.add_argument('--valid', type=int, default=1)
    parser.add_argument('--experiment', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--path', nargs='?', default='./datasets/')

    # Training tool
    parser.add_argument('--epochs', type=int, default=300)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    set_settings(args)
    log = Logger(args)
    args.log = log
    log(str(args))
    RunExperiments(log, args)



