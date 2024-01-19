# -*- coding: utf-8 -*-
# Author : yuxiang Zeng

import torch

from utils.metamodel import MetaModel


# 模型二
class NAS_Model_2(MetaModel):
    def __init__(self, args):
        super(NAS_Model_2, self).__init__(args)
        self.args = args
        self.dim = args.dimension
        self.ops_embeds = torch.nn.Linear(6, self.dim)
        self.host_embeds = torch.nn.Embedding(20, self.dim)

        input_dim = 2 * self.dim
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
        torch.nn.init.kaiming_normal_(self.ops_embeds.weight)
        torch.nn.init.kaiming_normal_(self.host_embeds.weight)

    def forward(self, inputs, train = True):
        firstIdx, secondIdx, thirdIdx, fourthIdx, fifthIdx, sixthIdx, seventhIdx = self.get_inputs(inputs)
        # print(seventhIdx.shape)
        x = torch.stack([secondIdx, thirdIdx, fourthIdx, fifthIdx, sixthIdx, seventhIdx], dim=-1).float()
        ops_embeds = self.ops_embeds(x)
        host_embeds = self.host_embeds(firstIdx)
        estimated = self.NeuCF(
            torch.cat((ops_embeds, host_embeds),
                      dim=-1)).sigmoid().reshape(-1)
        return estimated

    def prepare_test_model(self):
        pass

    def get_inputs(self, inputs):
        firstIdx, secondIdx, thirdIdx, fourthIdx, fifthIdx, sixthIdx, seventhIdx = inputs
        return firstIdx.long(), secondIdx.long(), thirdIdx.long(), fourthIdx.long(), fifthIdx.long(), sixthIdx.long(), seventhIdx.long()

