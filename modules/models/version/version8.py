# -*- coding: utf-8 -*-
# Author : yuxiang Zeng

import dgl
import torch
from utils.metamodel import MetaModel
from dgl.nn.pytorch import GraphConv, SAGEConv

# 优化到极致的GNN
class NAS_Model_Chatgpt_GNN_3(MetaModel):
    def __init__(self, args):
        super(NAS_Model_Chatgpt_GNN_3, self).__init__(args)
        self.dim = args.dimension
        #
        self.platform_embeds = torch.nn.Embedding(6, self.dim)
        self.device_embeds = torch.nn.Embedding(6, self.dim)
        self.device_name_embeds = torch.nn.Embedding(6, self.dim)
        self.precision_embeds = torch.nn.Embedding(6, self.dim)

        # 第二个想法
        if args.llm:
            self.transfer = torch.nn.Linear(5, self.dim)
        else:
            self.transfer = torch.nn.Linear(1, self.dim)

        self.op_embeds = torch.nn.Embedding(6, self.dim)

        input_dim = 12 * self.dim
        self.NeuCF = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim // 2),  # FFN
            torch.nn.LayerNorm(input_dim // 2),  # LayerNorm
            torch.nn.ReLU(),  # ReLU
            torch.nn.Linear(input_dim // 2, input_dim // 2),  # FFN
            torch.nn.LayerNorm(input_dim // 2),  # LayerNorm
            torch.nn.ReLU(),  # ReLU
            torch.nn.Linear(input_dim // 2, 1)  # y
        )
        self.initialize()
        self.cache = {}

    def initialize(self):
        pass

    def forward(self, inputs, train=True):
        if self.args.llm:
            platformIdx, deviceIdx, device_info_llm, precisionIdx, op_idx = self.get_inputs(inputs)
        else:
            platformIdx, deviceIdx, device_name_Idx, precisionIdx, op_idx = self.get_inputs2(inputs)

        # DNN network
        firstIdx = op_idx[:, 0]
        secondIdx = op_idx[:, 1]
        thirdIdx = op_idx[:, 2]
        fourthIdx = op_idx[:, 3]
        fifthIdx = op_idx[:, 4]
        sixthIdx = op_idx[:, 5]
        seventhIdx = op_idx[:, 6]
        eighthIdx = op_idx[:, 7]

        platform_embeds = self.platform_embeds(platformIdx)
        device_embeds = self.device_embeds(deviceIdx)

        if self.args.llm:
            device_name_embeds = self.transfer(device_info_llm)
        else:
            # device_name_embeds = self.device_name_embeds(device_name_Idx)
            device_name_embeds = self.transfer(device_name_Idx.reshape(-1, 1).float())

        precision_embeds = self.precision_embeds(precisionIdx)

        # DNN
        first_embeds = self.op_embeds(firstIdx)
        second_embeds = self.op_embeds(secondIdx)
        third_embeds = self.op_embeds(thirdIdx)
        fourth_embeds = self.op_embeds(fourthIdx)
        fifth_embeds = self.op_embeds(fifthIdx)
        sixth_embeds = self.op_embeds(sixthIdx)
        seventh_embeds = self.op_embeds(seventhIdx)
        eighth_embeds = self.op_embeds(eighthIdx)

        # Final interaction
        final_input = torch.cat([
            platform_embeds, device_embeds, device_name_embeds, precision_embeds,
            first_embeds, second_embeds, third_embeds, fourth_embeds,
            fifth_embeds, sixth_embeds, seventh_embeds, eighth_embeds
        ], dim=-1)
        estimated = self.NeuCF(final_input).sigmoid().reshape(-1)
        return estimated

    def prepare_test_model(self):
        pass

    def get_inputs(self, inputs):
        platformIdx, deviceIdx, precisionIdx, \
            frequency, cores, threads, memory_size, memory_speed, \
            fifthIdx, sixthIdx, seventhIdx, eighthIdx, ninthIdx, tenthIdx, elemIdx = inputs

        # firstIdx, secondIdx, thirdIdx, fourthIdx,\
        op_idx = torch.vstack([fifthIdx, sixthIdx, seventhIdx, eighthIdx, ninthIdx, tenthIdx, elemIdx])
        insert_back = 4 * torch.ones(1, op_idx.shape[1])
        # 添加输入输出节点
        op_idx = torch.cat([op_idx, insert_back]).transpose(0, 1)
        op_idx = op_idx.to(torch.long)
        device_info_llm = torch.vstack([frequency, cores, threads, memory_size, memory_speed]).T
        # return platformIdx.long(), deviceIdx.long(), device_name_Idx.long(), precisionIdx.long(), op_idx.long()
        return platformIdx.long(), deviceIdx.long(), device_info_llm.float(), precisionIdx.long(), op_idx.long(),


    def get_inputs2(self, inputs):
        firstIdx, secondIdx, thirdIdx, fourthIdx, \
            fifthIdx, sixthIdx, seventhIdx, eighthIdx, ninthIdx, tenthIdx, elemIdx = inputs
        op_idx = torch.vstack([fifthIdx, sixthIdx, seventhIdx, eighthIdx, ninthIdx, tenthIdx, elemIdx])
        insert_back = 4 * torch.ones(1, op_idx.shape[1])
        # 添加输入输出节点
        op_idx = torch.cat([op_idx, insert_back]).transpose(0, 1)
        op_idx = op_idx.to(torch.long)
        # 获得计算节点信息
        platformIdx = firstIdx
        deviceIdx = secondIdx
        device_name_Idx = thirdIdx
        precisionIdx = fourthIdx

        return platformIdx.long(), deviceIdx.long(), device_name_Idx.long(), precisionIdx.long(), op_idx.long()

