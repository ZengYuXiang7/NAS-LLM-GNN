# -*- coding: utf-8 -*-
# Author : yuxiang Zeng

import torch

from utils.metamodel import MetaModel


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
        torch.nn.init.kaiming_normal_(self.first_embeds.weight)
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
