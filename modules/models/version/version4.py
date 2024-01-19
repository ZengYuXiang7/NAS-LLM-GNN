# -*- coding: utf-8 -*-
# Author : yuxiang Zeng

import torch

from utils.metamodel import MetaModel


# 模型四
class NAS_Model_4(MetaModel):
    def __init__(self, args):
        super(NAS_Model_4, self).__init__(args)
        self.dim = args.dimension

        # 4, 4, 12, 4
        self.platform_embeds = torch.nn.Embedding(4, self.dim)
        self.device_embeds = torch.nn.Embedding(4, self.dim)
        self.devices_id_embeds = torch.nn.Embedding(12, self.dim)
        self.precision_embeds = torch.nn.Embedding(4, self.dim)

        #
        self.first_embeds = torch.nn.Embedding(5, self.dim)
        self.second_embeds = torch.nn.Embedding(5, self.dim)
        self.third_embeds = torch.nn.Embedding(5, self.dim)
        self.fourth_embeds = torch.nn.Embedding(5, self.dim)
        self.fifth_embeds = torch.nn.Embedding(5, self.dim)
        self.sixth_embeds = torch.nn.Embedding(5, self.dim)

        input_dim = 10 * self.dim
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
        torch.nn.init.kaiming_normal_(self.platform_embeds.weight)
        torch.nn.init.kaiming_normal_(self.device_embeds.weight)
        torch.nn.init.kaiming_normal_(self.devices_id_embeds.weight)
        torch.nn.init.kaiming_normal_(self.precision_embeds.weight)

        torch.nn.init.kaiming_normal_(self.first_embeds.weight)
        torch.nn.init.kaiming_normal_(self.second_embeds.weight)
        torch.nn.init.kaiming_normal_(self.third_embeds.weight)
        torch.nn.init.kaiming_normal_(self.fourth_embeds.weight)
        torch.nn.init.kaiming_normal_(self.fifth_embeds.weight)
        torch.nn.init.kaiming_normal_(self.sixth_embeds.weight)

    def forward(self, inputs, train = True):
        platformIdx, deviceIdx, devices_idIdx, precisionIdx, firstIdx, secondIdx, thirdIdx, fourthIdx, fifthIdx, sixthIdx = self.get_inputs(
            inputs)

        platform_embeds = self.platform_embeds(platformIdx)
        device_embeds = self.device_embeds(deviceIdx)
        devices_id_embeds = self.devices_id_embeds(devices_idIdx)
        precision_embeds = self.precision_embeds(precisionIdx)

        first_embeds = self.first_embeds(firstIdx)
        second_embeds = self.second_embeds(secondIdx)
        third_embeds = self.third_embeds(thirdIdx)
        fourth_embeds = self.fourth_embeds(fourthIdx)
        fifth_embeds = self.fifth_embeds(fifthIdx)
        sixth_embeds = self.sixth_embeds(sixthIdx)

        estimated = self.NeuCF(
            torch.cat(
                (
                    platform_embeds, device_embeds, devices_id_embeds, precision_embeds,
                    first_embeds, second_embeds, third_embeds, fourth_embeds, fifth_embeds, sixth_embeds
                ), dim=-1)
        ).sigmoid().reshape(-1)

        return estimated

    def prepare_test_model(self):
        pass

    def get_inputs(self, inputs):
        platformIdx, deviceIdx, devices_idIdx, precisionIdx, firstIdx, secondIdx, thirdIdx, fourthIdx, fifthIdx, sixthIdx = inputs

        return platformIdx.long(), deviceIdx.long(), devices_idIdx.long(), precisionIdx.long(), \
            firstIdx.long(), secondIdx.long(), thirdIdx.long(), fourthIdx.long(), fifthIdx.long(), sixthIdx.long()

