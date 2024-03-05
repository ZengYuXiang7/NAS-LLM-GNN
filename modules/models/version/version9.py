# -*- coding: utf-8 -*-
# Author : yuxiang Zeng
import time

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

from utils.metrics import ErrorMetrics
from utils.trainer import get_loss_function, get_optimizer
from utils.utils import to_cuda, optimizer_step, lr_scheduler_step, optimizer_zero_grad


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True, weight_init='thomas', bias_init='thomas'):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.weight_init = weight_init
        self.bias_init = bias_init
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, adjacency, features):
        support = torch.matmul(features, self.weight)
        output = torch.bmm(adjacency, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GCN(Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.args = args
        num_features = 0
        num_layers = 2
        num_hidden = 32
        dropout_ratio = 0
        weight_init = 'thomas'
        bias_init = 'thomas'
        binary_classifier = False
        augments = 0

        self.nfeat = num_features
        self.nlayer = num_layers
        self.nhid = num_hidden
        self.dropout_ratio = dropout_ratio

        self.gc = nn.ModuleList([GraphConvolution(self.nfeat if i == 0 else self.nhid, self.nhid, bias=True, weight_init=weight_init, bias_init=bias_init) for i in range(self.nlayer)])
        self.bn = nn.ModuleList([nn.LayerNorm(self.nhid).double() for i in range(self.nlayer)])
        self.relu = nn.ModuleList([nn.ReLU().double() for i in range(self.nlayer)])
        if not binary_classifier:
            self.fc = nn.Linear(self.nhid + augments, 1).double()
        else:
            if binary_classifier == 'naive':
                self.fc = nn.Linear(self.nhid + augments, 1).double()
            elif binary_classifier == 'oneway' or binary_classifier == 'oneway-hard':
                self.fc = nn.Linear((self.nhid + augments) * 2, 1).double()
            else:
                self.fc = nn.Linear((self.nhid + augments) * 2, 2).double()

            if binary_classifier != 'oneway' and binary_classifier != 'oneway-hard':
                self.final_act = nn.LogSoftmax(dim=1)
            else:
                self.final_act = nn.Sigmoid()
        self.dropout = nn.ModuleList([nn.Dropout(self.dropout_ratio).double() for i in range(self.nlayer)])
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

    def forward(self, inputs, augments=None):
        adjacency, features = inputs
        augments = None
        if not self.binary_classifier:
            x = self.forward_single_model(adjacency, features)
            x = x[:, 0]  # use global node
            if augments is not None:
                x = torch.cat([x, augments], dim=1)
            return self.fc(x)
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
        if args.device != 'cpu':
            self.to(self.args.device)
            self.loss_function = get_loss_function(args).to(self.args.device)
        else:
            self.loss_function = get_loss_function(args)
        self.optimizer = get_optimizer(self.parameters(), lr=args.lr, decay=args.decay, args=args)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_step, gamma=0.50)


    def train_one_epoch(self, dataModule):
        loss = None
        self.train()
        torch.set_grad_enabled(True)
        t1 = time.time()
        for train_Batch in tqdm(dataModule.train_loader, disable=not self.args.program_test, ):
            inputs, value = train_Batch
            if self.args.device != 'cpu':
                inputs, value = to_cuda(inputs, value)
            pred = self.forward(inputs, True)
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
        preds = torch.zeros((len(dataModule.valid_loader.dataset),)).to('cuda') if self.args.device != 'cpu' else torch.zeros((len(dataModule.valid_loader.dataset),))
        reals = torch.zeros((len(dataModule.valid_loader.dataset),)).to('cuda') if self.args.device != 'cpu' else torch.zeros((len(dataModule.valid_loader.dataset),))
        self.prepare_test_model()
        for valid_Batch in tqdm(dataModule.valid_loader, disable=not self.args.program_test):
            inputs, value = valid_Batch
            if self.args.device != 'cpu':
                inputs, value = to_cuda(inputs, value)
            pred = self.forward(inputs, False)
            preds[writeIdx:writeIdx + len(pred)] = pred
            reals[writeIdx:writeIdx + len(value)] = value
            writeIdx += len(pred)
        # print(torch.max(reals), torch.max(preds))
        valid_error = ErrorMetrics(reals * dataModule.max_value, preds * dataModule.max_value)
        return valid_error

    def test_one_epoch(self, dataModule):
        writeIdx = 0
        preds = torch.zeros((len(dataModule.test_loader.dataset),)).to('cuda') if self.args.device != 'cpu' else torch.zeros((len(dataModule.test_loader.dataset),))
        reals = torch.zeros((len(dataModule.test_loader.dataset),)).to('cuda') if self.args.device != 'cpu' else torch.zeros((len(dataModule.test_loader.dataset),))
        self.prepare_test_model()
        for test_Batch in tqdm(dataModule.test_loader, disable=not self.args.program_test):
            inputs, value = test_Batch
            if self.args.device != 'cpu':
                inputs, value = to_cuda(inputs, value)
            pred = self.forward(inputs, False)
            preds[writeIdx:writeIdx + len(pred)] = pred
            reals[writeIdx:writeIdx + len(value)] = value
            writeIdx += len(pred)
        test_error = ErrorMetrics(reals * dataModule.max_value, preds * dataModule.max_value)
        return test_error

