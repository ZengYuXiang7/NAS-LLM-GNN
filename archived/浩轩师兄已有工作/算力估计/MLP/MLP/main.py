import pickle
import os
import numpy as np
import torch as t
from torch.nn import *
from torch.utils.data import Dataset, DataLoader
import copy
from tqdm import tqdm


class EarlyStopMonitor:

    def __init__(self, patient):
        self.model = None
        self.patient = patient
        self.counter = 0
        self.val = 1e10
        self.epoch = -1

    def early_stop(self):
        return self.counter >= self.patient

    def track(self, epoch, model, val):
        if val < self.val:
            self.model = copy.deepcopy(model)
            self.epoch = epoch
            self.val = val
            self.counter = 0
        else:
            self.counter += 1


class DriveDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _, node_atr, tnode_index, egde_atr = self.data[idx]
        return np.array(node_atr), tnode_index, egde_atr


def load_dataset():
    filePath = 'dataset/'
    arr = os.listdir(filePath)
    tmp = []
    model = dict()
    res = []
    for file in arr:
        data = pickle.load(open('./dataset/' + file, 'rb'))
        data = list(data.items())
        tmp.append(data)

    for i, v in enumerate(tmp):
        for j in v:
            tnode_index = i
            node_atr = j[0]
            egde_atr = j[1]
            if j[0] not in model:
                model[j[0]] = len(model)
            snode_index = model[j[0]]
            # print(snode_index)
            # print(node_atr)
            # print(tnode_index)
            # print(egde_atr)
            res.append([snode_index, node_atr, tnode_index, egde_atr])
    # exit()
    # print(np.array(res))
    return np.array(res)


def split_train_valid_test(dataset):
    perm = np.random.permutation(len(dataset))
    train_size = int(len(dataset[:, 0]) * 0.6)
    valid_size = int(len(dataset[:, 0]) * 0.2)

    trainIdx = perm[:train_size]
    validIdx = perm[train_size:train_size + valid_size]
    testIdx = perm[train_size + valid_size:]

    train_dataset = dataset[trainIdx]
    valid_dataset = dataset[validIdx]
    test_dataset = dataset[testIdx]

    return train_dataset, valid_dataset, test_dataset


def get_dataloaders():
    dataset = load_dataset()
    trainset, validset, testset = split_train_valid_test(dataset)
    trainLoader = DataLoader(DriveDataset(trainset), batch_size=128, shuffle=True)
    validLoader = DataLoader(DriveDataset(validset), batch_size=1024)
    testLoader = DataLoader(DriveDataset(testset), batch_size=1024)
    return trainLoader, validLoader, testLoader


class MLPPredictorV2(Module):

    def __init__(self, dim=32, device='cpu'):
        super(MLPPredictorV2, self).__init__()
        self.device = device
        self.ops_embeds = Linear(6, dim)
        self.host_embeds = Embedding(20, dim)
        self.score = Sequential(
            Linear(2 * dim, 2 * dim),
            ReLU(),
            BatchNorm1d(2 * dim),
            Linear(2 * dim, dim),
            ReLU(),
            BatchNorm1d(dim),
            Linear(dim, 1)
        )

    def forward(self, ops, host):
        # print(ops, host)
        # ops = [bs, num_ops]
        ops_embeds = self.ops_embeds(ops.float())
        # host = [N]
        host_embeds = self.host_embeds(host)
        # scoring
        input_feats = t.cat([ops_embeds, host_embeds], dim=-1)
        score = self.score(input_feats).squeeze()
        return score


class MLPPredictor(Module):

    def __init__(self, dim=32, device='cpu'):
        super(MLPPredictor, self).__init__()
        self.device = device
        self.op_embeds = Embedding(10, dim)
        self.host_embeds = Embedding(20, dim)
        self.rnn = GRU(dim, dim, num_layers=1, batch_first=True)
        self.score = Sequential(
            Linear(2 * dim, 2 * dim),
            ReLU(),
            BatchNorm1d(2 * dim),
            Linear(2 * dim, dim),
            ReLU(),
            BatchNorm1d(dim),
            Linear(dim, 1)
        )

    def forward(self, ops, host):
        # ops = [bs, num_ops]
        ops_embeds = self.op_embeds(ops)
        print(ops_embeds.shape)
        _, ops_embeds = self.rnn(ops_embeds)
        ops_embeds = ops_embeds.squeeze()

        # host = [N]
        host_embeds = self.host_embeds(host)

        # scoring
        input_feats = t.cat([ops_embeds, host_embeds], dim=-1)
        score = self.score(input_feats).squeeze()
        return score


def ErrMetrics(pred, true):
    nonzeroIdx = true.nonzero()
    true = true[nonzeroIdx]
    pred = pred[nonzeroIdx]
    ER = np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum(true ** 2))
    NMAE = np.sum(np.abs(true - pred)) / np.sum(true)
    return ER, NMAE


def model_evaluation(model, validLoader):
    writeIdx = 0
    preds = t.zeros((len(validLoader.dataset),)).to(model.device)
    reals = t.zeros((len(validLoader.dataset),)).to(model.device)
    model.eval()
    t.set_grad_enabled(False)
    for validBatch in validLoader:
        ops, host, target = validBatch
        pred = model.forward(ops, host)
        preds[writeIdx:writeIdx + len(pred)] = pred.cpu()
        reals[writeIdx:writeIdx + len(pred)] = target.float()
        writeIdx += len(pred)
    NRMSE, NMAE = ErrMetrics(preds.numpy(), reals.numpy())
    t.set_grad_enabled(True)
    return NRMSE, NMAE


if __name__ == '__main__':

    trainLoader, validLoader, testLoader = get_dataloaders()
    criterion = MSELoss()
    # model = MLPPredictorV2()
    model = MLPPredictor()
    optimizer = t.optim.Adam(model.parameters())
    monitor = EarlyStopMonitor(10)

    for epoch in range(50):
        for trainBatch in tqdm(trainLoader):
            optimizer.zero_grad()
            ops, host, target = trainBatch
            pred = model.forward(ops, host)
            loss = criterion(target.float(), pred)
            loss.backward()
            optimizer.step()

        vNRMSE, vNMAE = model_evaluation(model, validLoader)
        print(f'Epoch={epoch}, vNRMSE={vNRMSE:.3f}, vNMAE={vNMAE:.3f}')
        monitor.track(epoch, model, vNRMSE)
        if monitor.early_stop():
            break

    tNRMSE, tNMAE = model_evaluation(monitor.model, testLoader)
    print(f'Experiment Done! tNRMSE={tNRMSE:.3f}, tNMAE={tNMAE:.3f}')
