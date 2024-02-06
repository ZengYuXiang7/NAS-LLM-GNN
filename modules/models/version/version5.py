# -*- coding: utf-8 -*-
# Author : yuxiang Zeng

import torch

from utils.metamodel import MetaModel


# 模型五
class NAS_Model_Chatgpt(MetaModel):
    def __init__(self, args):
        super(NAS_Model_Chatgpt, self).__init__(args)
        self.dim = args.dimension

        # 4, 4, 12, 4
        self.platform_embeds = torch.nn.Embedding(4, self.dim)
        self.device_embeds = torch.nn.Embedding(4, self.dim)
        self.precision_embeds = torch.nn.Embedding(4, self.dim)

        # Device_more_info
        self.info_embeds = torch.nn.Sequential(
            torch.nn.Linear(5, self.dim),
            torch.nn.LayerNorm(self.dim),  # LayerNorm
            torch.nn.ReLU(),               # ReLU
            torch.nn.Linear(self.dim, self.dim),
        )

        self.op_embeds = torch.nn.Embedding(6, self.dim)

        input_dim = 10 * self.dim
        # input_dim = 4 * self.dim
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
        torch.nn.init.kaiming_normal_(self.precision_embeds.weight)

        torch.nn.init.kaiming_normal_(self.first_embeds.weight)
        torch.nn.init.kaiming_normal_(self.second_embeds.weight)
        torch.nn.init.kaiming_normal_(self.third_embeds.weight)
        torch.nn.init.kaiming_normal_(self.fourth_embeds.weight)
        torch.nn.init.kaiming_normal_(self.fifth_embeds.weight)
        torch.nn.init.kaiming_normal_(self.sixth_embeds.weight)

    def forward(self, inputs, train = True):
        platformIdx, deviceIdx, precisionIdx, \
            frequency, cores, threads, memory_size, memory_speed, \
            firstIdx, secondIdx, thirdIdx, fourthIdx, fifthIdx, sixthIdx = self.get_inputs(inputs)

        # Device
        platform_embeds = self.platform_embeds(platformIdx)
        device_embeds = self.device_embeds(deviceIdx)
        precision_embeds = self.precision_embeds(precisionIdx)

        # Device more info
        device_info = torch.stack([frequency, cores, threads, memory_size, memory_speed], dim=-1)
        # device_info = torch.stack([frequency, threads, memory_size, memory_speed], dim=-1)
        device_features = self.info_embeds(device_info)

        # DNN network
        first_embeds = self.op_embeds(firstIdx)
        second_embeds = self.op_embeds(secondIdx)
        third_embeds = self.op_embeds(thirdIdx)
        fourth_embeds = self.op_embeds(fourthIdx)
        fifth_embeds = self.op_embeds(fifthIdx)
        sixth_embeds = self.op_embeds(sixthIdx)

        # print(platform_embeds.shape)
        # print(device_info.shape)

        final_input = torch.cat([
            platform_embeds, device_embeds, precision_embeds,
            device_features,  # device_info,
            first_embeds, second_embeds, third_embeds, fourth_embeds, fifth_embeds, sixth_embeds
        ], dim=-1)

        # print(final_input.shape)
        estimated = self.NeuCF(final_input).sigmoid().reshape(-1)

        return estimated

    def prepare_test_model(self):
        pass

    @staticmethod
    def get_inputs(inputs):
        platformIdx, deviceIdx, precisionIdx, \
            frequency, cores, threads, memory_size, memory_speed, \
            firstIdx, secondIdx, thirdIdx, fourthIdx, fifthIdx, sixthIdx = inputs
        # print(frequency, cores, threads, memory_size, memory_speed)
        return platformIdx.long(), deviceIdx.long(), precisionIdx.long(), \
            frequency.float(), cores.float(), threads.float(), memory_size.float(), memory_speed.float(), \
            firstIdx.long(), secondIdx.long(), thirdIdx.long(), fourthIdx.long(), fifthIdx.long(), sixthIdx.long()

