# -*- coding: utf-8 -*-
# Author : yuxiang Zeng

import dgl
import torch
from utils.metamodel import MetaModel
from dgl.nn.pytorch import GraphConv, SAGEConv


# 优化到极致的GNN
class GraphSAGEConv(torch.nn.Module):
    def __init__(self, dim, order=2):
        super(GraphSAGEConv, self).__init__()
        self.order = order
        self.layers = torch.nn.ModuleList([SAGEConv(dim, dim, aggregator_type='gcn') for _ in range(order)])
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(dim) for _ in range(order)])
        self.acts = torch.nn.ModuleList([torch.nn.ELU() for _ in range(order)])

    def forward(self, graph, features):
        g, g.ndata['L0'] = graph, features
        feats = g.ndata['L0']
        for i, (layer, norm, act) in enumerate(zip(self.layers, self.norms, self.acts)):
            feats = layer(g, feats).squeeze()
            feats = norm(feats)
            feats = act(feats)
            g.ndata[f'L{i + 1}'] = feats
        embeds = g.ndata[f'L{self.order}']
        return embeds


class ReadoutLayer(torch.nn.Module):
    def forward(self, batched_g, node_embeddings, op_idx):
        batched_g.ndata['h'] = node_embeddings
        # 对每个图进行求平均
        hg_list = []
        cum_node_count = 0
        for i, num_nodes in enumerate(batched_g.batch_num_nodes()):
            nodes = torch.arange(cum_node_count, cum_node_count + num_nodes, device=node_embeddings.device)
            cum_node_count += num_nodes
            mask = (op_idx[i] != 5)  # 将 op_idx 扁平化并创建掩码
            nodes = nodes[mask == True]
            hg = batched_g.ndata['h'][nodes].mean(dim=0)
            hg_list.append(hg)
        hg = torch.stack(hg_list)
        return hg


class NAS_Model_Chatgpt_GNN_2(MetaModel):
    def __init__(self, args):
        super(NAS_Model_Chatgpt_GNN_2, self).__init__(args)
        self.dim = args.dimension
        # 4, 4, 12, 4
        self.platform_embeds = torch.nn.Embedding(4, self.dim)
        self.device_embeds = torch.nn.Embedding(4, self.dim)
        self.precision_embeds = torch.nn.Embedding(4, self.dim)

        # Device_more_info
        self.info_embeds = torch.nn.Sequential(
            torch.nn.Linear(5, self.dim),
            torch.nn.LayerNorm(self.dim),  # LayerNorm
            torch.nn.ReLU(),  # ReLU
            torch.nn.Linear(self.dim, self.dim),
        )

        # 弃用随机初始化
        # self.op_embeds = torch.nn.Embedding(6, self.dim)
        # self.op_transfer = torch.nn.Linear(self.dim, self.dim)
        # 0 con1 1 con3 2 max3 3 input 4 output 5 None
        self.op_embeds = torch.tensor([0.391, 1.000, 0.318, 0.004, 0.035, 0.0])
        self.op_transfer = torch.nn.Linear(1, self.dim)
        self.gnn = GraphSAGEConv(self.dim, 2)
        self.readout = ReadoutLayer()

        input_dim = 4 * self.dim + 1 * self.dim
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
        torch.nn.init.kaiming_normal_(self.platform_embeds.weight)
        torch.nn.init.kaiming_normal_(self.device_embeds.weight)
        torch.nn.init.kaiming_normal_(self.precision_embeds.weight)

    def forward(self, inputs, train=True):
        platformIdx, deviceIdx, precisionIdx, \
            frequency, cores, threads, memory_size, memory_speed, \
            op_idx, graph = self.get_inputs(inputs)

        # Device
        platform_embeds = self.platform_embeds(platformIdx)
        device_embeds = self.device_embeds(deviceIdx)
        precision_embeds = self.precision_embeds(precisionIdx)

        # Device more info
        device_info = torch.stack([frequency, cores, threads, memory_size, memory_speed], dim=-1)
        device_features = self.info_embeds(device_info)

        # DNN network
        # op_embeds = self.op_embeds(op_idx)
        op_embeds = self.op_embeds[op_idx].reshape(-1, 8, 1)
        op_embeds = self.op_transfer(op_embeds)
        # 形状为 [32, 8, dim]
        op_embeds = op_embeds.view(-1, self.dim)  # 重塑为 [32*8, dim]
        dnn_embeds = self.gnn(graph, op_embeds)
        dnn_embeds = self.readout(graph, dnn_embeds, op_idx).reshape(-1, 1, self.dim)

        # Final interaction
        final_input = torch.cat([
            platform_embeds, device_embeds, precision_embeds,
            device_features,  # device_info,
            dnn_embeds
        ], dim=-1)

        estimated = self.NeuCF(final_input).sigmoid().reshape(-1)

        return estimated

    def prepare_test_model(self):
        pass

    def get_inputs(self, inputs):
        op, graph = inputs
        op_idx = op[:, 8: -1]

        insert_front = 3.
        insert_back = 4.

        # 添加输入输出节点
        if self.args.device != 'cpu':
            op_idx = torch.stack(
                [torch.cat(
                    (torch.tensor([insert_front]).to('cuda'), row, torch.tensor([insert_back]).to('cuda'))
                ) for row in op_idx]
            )
        else:
            op_idx = torch.stack(
                [torch.cat(
                    (torch.tensor([insert_front]), row, torch.tensor([insert_back]))
                ) for row in op_idx]
            )
        op_idx = op_idx.to(torch.long)

        # 获得计算节点信息
        op = op[:, :8]
        platformIdx = op[:, 0:1]
        deviceIdx = op[:, 1:2]
        precisionIdx = op[:, 2:3]
        frequency = op[:, 3:4]
        cores = op[:, 4:5]
        threads = op[:, 5:6]
        memory_size = op[:, 6:7]
        memory_speed = op[:, 7:8]

        return platformIdx.long(), deviceIdx.long(), precisionIdx.long(), \
            frequency.float(), cores.float(), threads.float(), memory_size.float(), memory_speed.float(), \
            op_idx.long(), graph
