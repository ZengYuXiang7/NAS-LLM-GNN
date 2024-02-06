# -*- coding: utf-8 -*-
# Author : yuxiang Zeng


import time
import torch

from tqdm import *
from abc import ABC, abstractmethod

from utils.metrics import ErrorMetrics
from utils.trainer import get_loss_function, get_optimizer
from utils.utils import optimizer_zero_grad, optimizer_step, lr_scheduler_step, to_cuda


class MetaModel(torch.nn.Module, ABC):
    def __init__(self, args):
        super(MetaModel, self).__init__()
        self.args = args

    @abstractmethod
    def forward(self, inputs, train = True):
        pass

    @abstractmethod
    def prepare_test_model(self):
        pass

    def set_epochs(self, epochs):
        self.epochs = epochs

    def setup_optimizer(self, args):
        if args.device != 'cpu':
            self.to(self.args.device)
            self.loss_function = get_loss_function(args).to(self.args.device)
        else:
            self.loss_function = get_loss_function(args)
        self.optimizer = get_optimizer(self.parameters(), lr=args.lr, decay=args.decay, args=args)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, threshold=0.01)


    def train_one_epoch(self, dataModule):
        loss = None
        self.train()
        torch.set_grad_enabled(True)
        t1 = time.time()
        for train_Batch in tqdm(dataModule.train_loader, disable=not self.args.program_test, ):
            if self.args.exper not in [6, 7]:
                inputs, value = train_Batch
            else:
                op, graph, value = train_Batch
                inputs = op, graph
            if self.args.device != 'cpu':
                inputs, value = to_cuda(inputs, value)
            pred = self.forward(inputs, True)
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
        preds = torch.zeros((len(dataModule.valid_loader.dataset),)).to('cuda') if self.args.device != 'cpu' else torch.zeros((len(dataModule.valid_loader.dataset),))
        reals = torch.zeros((len(dataModule.valid_loader.dataset),)).to('cuda') if self.args.device != 'cpu' else torch.zeros((len(dataModule.valid_loader.dataset),))
        self.prepare_test_model()
        for valid_Batch in tqdm(dataModule.valid_loader, disable=not self.args.program_test):
            if self.args.exper not in [6, 7]:
                inputs, value = valid_Batch
            else:
                op, graph, value = valid_Batch
                inputs = op, graph
            if self.args.device != 'cpu':
                inputs, value = to_cuda(inputs, value)
            pred = self.forward(inputs, False)
            preds[writeIdx:writeIdx + len(pred)] = pred
            reals[writeIdx:writeIdx + len(value)] = value
            writeIdx += len(pred)
            val_loss += self.loss_function(pred, value).item()
        self.scheduler.step(val_loss)
        valid_error = ErrorMetrics(reals * dataModule.max_value, preds * dataModule.max_value)
        return valid_error

    def test_one_epoch(self, dataModule):
        writeIdx = 0
        preds = torch.zeros((len(dataModule.test_loader.dataset),)).to('cuda') if self.args.device != 'cpu' else torch.zeros((len(dataModule.test_loader.dataset),))
        reals = torch.zeros((len(dataModule.test_loader.dataset),)).to('cuda') if self.args.device != 'cpu' else torch.zeros((len(dataModule.test_loader.dataset),))
        self.prepare_test_model()
        for test_Batch in tqdm(dataModule.test_loader, disable=not self.args.program_test):
            if self.args.exper not in [6, 7]:
                inputs, value = test_Batch
            else:
                op, graph, value = test_Batch
                inputs = op, graph
            if self.args.device != 'cpu':
                inputs, value = to_cuda(inputs, value)
            pred = self.forward(inputs, False)
            preds[writeIdx:writeIdx + len(pred)] = pred
            reals[writeIdx:writeIdx + len(value)] = value
            writeIdx += len(pred)
        test_error = ErrorMetrics(reals * dataModule.max_value, preds * dataModule.max_value)
        return test_error
