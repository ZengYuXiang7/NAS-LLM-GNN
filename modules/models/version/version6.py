# -*- coding: utf-8 -*-
# Author : yuxiang Zeng

import torch
import dgl
from utils.metamodel import MetaModel


# 定义图卷积网络层
class GraphConvolution(torch.nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GraphConvolution, self).__init__()
        self.linear = torch.nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(dgl.function.copy_u('h', 'm'), dgl.function.sum('m', 'h'))
            h = g.ndata['h']
            return self.linear(h)

# 定义GNN模型
class GNN(torch.nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(GNN, self).__init__()
        self.conv1 = GraphConvolution(in_feats, hidden_size)
        self.conv2 = GraphConvolution(hidden_size, out_feats)

    def forward(self, g, features):
        out_feats = []
        for i in range(len(g)):
            x = torch.relu(self.conv1(g[i], features[i]))
            x = self.conv2(g[i], x)
            out_feats.append(x)
        out_feats = torch.stack(out_feats)
        return out_feats

# 定义Readout层
class ReadoutLayer(torch.nn.Module):
    def forward(self, g, node_embeddings):
        out_feats = []
        for i in range(len(g)):
            with g[i].local_scope():
                g[i].ndata['h'] = node_embeddings[i]
                hg = dgl.mean_nodes(g[i], 'h')
                out_feats.append(hg)
        out_feats = torch.stack(out_feats)
        return out_feats

class NAS_Model_Chatgpt_GNN(MetaModel):
    def __init__(self, args):
        super(NAS_Model_Chatgpt_GNN, self).__init__(args)
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
        self.gnn = GNN(self.dim, self.dim * 2, self.dim)
        self.readout = ReadoutLayer()

        input_dim = 5 * self.dim
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


    def forward(self, inputs, train = True):
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
        op_embeds = self.op_embeds(op_idx)
        dnn_embeds = self.gnn(graph, op_embeds)
        dnn_embeds = self.readout(graph, dnn_embeds)

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
