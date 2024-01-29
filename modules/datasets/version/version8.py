# coding : utf-8
# Author : yuxiang Zeng
import pickle
import torch
import dgl
from tqdm import *
from modules.datasets.chatgpt import NAS_ChatGPT
from utils.utils import *
import copy


# 图神经网络
class experiment8:
    def __init__(self, args):
        self.args = args
        self.chatgpt = NAS_ChatGPT(args)

    def get_device_item(self, file_names):
        import re, pandas
        from sklearn.preprocessing import LabelEncoder
        pattern = re.compile(r'^(?P<device_type>\w+)-(?P<unit>\w+)-(?P<devices>[\w-]+)-(?P<precision>\w+).pickle$')
        data = []
        for file_name in file_names:
            match = pattern.match(file_name)
            if match:
                data.append(match.groupdict())
        df = pandas.DataFrame(data).to_numpy()
        # 对每一列进行标签编码
        label_encoder = LabelEncoder()
        encoded_data = np.apply_along_axis(label_encoder.fit_transform, axis=0, arr=df)
        final_data = np.concatenate([encoded_data], axis=1)
        return final_data

    # 只是读取大文件
    def load_data(self, args):
        import os
        file_names = os.listdir(args.path)
        pickle_files = [file for file in file_names if file.endswith('.pickle')]
        device_label = self.get_device_item(pickle_files)
        data = []
        for i in range(len(pickle_files)):
            pickle_file = args.path + pickle_files[i]
            with open(pickle_file, 'rb') as f:
                now = pickle.load(f)
            data.append([now])
        data = np.array(data)
        data = np.concatenate([device_label, data], axis=1)
        return data

    def get_idx(self, op_seq):
        # 全部代码
        op_seq = list(op_seq)
        matrix, label = get_matrix_and_ops(op_seq)
        matrix, features = get_adjacency_and_features(matrix, label)

        def get_op_idx(features):
            result = [row.index(1) if 1 in row else 5 for row in features]
            return np.array(result)
        op_idx = get_op_idx(features)
        """
            本人定义：0 con1 1 con3 2 max3 3 input 4 output 5 None
            数据集定义：
                0 : None 5
                1 : None 5
                2 ： 0  con1
                3 ： 1  con2
                4 ： 2  max3
                input : 3
                output : 4
                [0, 1, 2, 3, 4, 5]
                [5, 5, 0, 1, 2, 3]
        """
        # print(op_seq)
        # print(label)
        # print(op_idx)
        # print('-' * 100)
        return op_idx

    def preprocess_data(self, data, args):
        try:
            tensor = pickle.load(open(f'./pretrained/tensor_{args.dataset_type}_{torch.initial_seed()}.pkl', 'rb'))
        except:
            tensor = []
            for i in trange(len(data)):
                for key in (data[i][-1].keys()):
                    now = []
                    # 添加设备号
                    for j in range(len(data[0]) - 1):
                        now.append(data[i][j])
                    # 添加模型号 转成的图 和 idx
                    op_idx = self.get_idx(key)
                    # 4 - 10 为 op_idx
                    for j in range(len(op_idx)):
                        now.append(op_idx[j])
                    y = data[i][-1][key]
                    now.append(y)
                    tensor.append(now)
            tensor = np.array(tensor)
        return tensor

    def get_pytorch_index(self, data):
        return torch.as_tensor(data)


def get_arch_vector_from_arch_str(arch_str):
    """
        Args:
            arch_str : a string representation of a cell architecture,
                for example '|nor_conv_3x3~0|+|nor_conv_3x3~0|avg_pool_3x3~1|+|skip_connect~0|nor_conv_3x3~1|skip_connect~2|'
    """
    _opname_to_index = {
        'none': 0,
        'skip_connect': 1,
        'nor_conv_1x1': 2,
        'nor_conv_3x3': 3,
        'avg_pool_3x3': 4,
        'input': 5,
        'output': 6,
        'global': 7
    }

    _opindex_to_name = {value: key for key, value in _opname_to_index.items()}
    nodes = arch_str.split('+')
    nodes = [node[1:-1].split('|') for node in nodes]
    nodes = [[op_and_input.split('~')[0] for op_and_input in node] for node in nodes]

    # arch_vector is equivalent to a decision vector produced by autocaml when using Nasbench201 backend
    arch_vector = [_opname_to_index[op] for node in nodes for op in node]
    return arch_vector


def get_arch_str_from_arch_vector(arch_vector):
    _opname_to_index = {
        'none': 0,
        'skip_connect': 1,
        'nor_conv_1x1': 2,
        'nor_conv_3x3': 3,
        'avg_pool_3x3': 4,
        'input': 5,
        'output': 6,
        'global': 7
    }
    _opindex_to_name = {value: key for key, value in _opname_to_index.items()}
    ops = [_opindex_to_name[opindex] for opindex in arch_vector]
    return '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.format(*ops)


def get_matrix_and_ops(g, prune=True, keep_dims=True):
    """
    返回邻接矩阵和节点标签。
    Args:
        g : 来自 Nasbench102 搜索空间的一个点（图结构）（列表形式）
        prune : 是否删除只连接到零操作的悬挂节点，默认为 True
        keep_dims : 在剪枝后是否保持原始矩阵大小，默认为 False
    """
    # 初始化8x8的邻接矩阵，所有元素都设为0
    matrix = [[0 for _ in range(8)] for _ in range(8)]

    # 初始化节点标签列表，初始时所有节点标签都为 None
    labels = [None for _ in range(8)]

    # 设置输入节点和输出节点的标签
    labels[0] = 'input'
    labels[-1] = 'output'

    # 初始化部分邻接矩阵的连接关系
    matrix[0][1] = matrix[0][2] = matrix[0][4] = 1
    matrix[1][3] = matrix[1][5] = 1
    matrix[2][6] = 1
    matrix[3][6] = 1
    matrix[4][7] = 1
    matrix[5][7] = 1
    matrix[6][7] = 1

    # 遍历输入的图结构
    for idx, op in enumerate(g):
        # 如果操作是零
        if op == 0:
            # 删除与该节点连接的边
            for other in range(8):  # 将这该序列点与全部节点断边
                # print(other, idx + 1)
                if matrix[other][idx + 1]:
                    matrix[other][idx + 1] = 0
                if matrix[idx + 1][other]:
                    matrix[idx + 1][other] = 0
        # 如果操作是跳跃连接
        elif op == 1:
            # 处理跳跃连接的情况
            to_del = []
            for other in range(8):
                # 若有谁连接到我，定点到这个谁
                if matrix[other][idx + 1]:
                    for other2 in range(8):
                        # 若我也连接到别人
                        if matrix[idx + 1][other2]:
                            # 就把连接到我的直接连到我连接的下一位
                            matrix[other][other2] = 1
                            # 断掉别人连向我的边
                            matrix[other][idx + 1] = 0
                            to_del.append(other2)

            # 删掉这些边
            for d in to_del:
                matrix[idx + 1][d] = 0
        # 如果操作是其他数字
        else:
            # 设置节点标签为操作的字符串表示
            labels[idx + 1] = str(op)

    # 如果选择剪枝操作
    if prune:
        # 初始化前向和后向访问标记列表
        visited_fw = [False for _ in range(8)]
        visited_bw = copy.copy(visited_fw)

        # 定义广度优先搜索函数
        def bfs(beg, vis, con_f):
            q = [beg]
            vis[beg] = True
            while q:
                v = q.pop()
                for other in range(8):
                    if not vis[other] and con_f(v, other):
                        q.append(other)
                        vis[other] = True

        # 前向搜索
        bfs(0, visited_fw, lambda src, dst: matrix[src][dst])
        # 后向搜索
        bfs(7, visited_bw, lambda src, dst: matrix[dst][src])

        # 遍历节点，删除未访问到的节点
        for v in range(7, -1, -1):
            if not visited_fw[v] or not visited_bw[v]:
                labels[v] = None
                # 如果选择保持原始矩阵大小
                if keep_dims:
                    matrix[v] = [0] * 8
                else:
                    del matrix[v]
                for other in range(len(matrix)):
                    # 如果选择保持原始矩阵大小
                    if keep_dims:
                        matrix[other][v] = 0
                    else:
                        del matrix[other][v]

        # 如果不保持原始矩阵大小
        if not keep_dims:
            # 移除标签中的 None 元素
            labels = list(filter(lambda l: l is not None, labels))

        # 断言保证首尾节点被访问到
        assert visited_fw[-1] == visited_bw[0]
        # 断言保证至少存在一个连接或者矩阵非空
        assert visited_fw[-1] == False or matrix

        # 断言确保矩阵的维度正确
        verts = len(matrix)
        assert verts == len(labels)
        for row in matrix:
            assert len(row) == verts

    # 返回最终邻接矩阵和节点标签
    return matrix, labels


def get_adjacency_and_features(matrix, labels):
    # 添加全局节点
    # for row in matrix:
    #     row.insert(0, 0)
    # global_row = [0, 1, 1, 1, 1, 1, 1, 1, 1]
    # matrix.insert(0, global_row)
    # # 添加对角线矩阵
    # for idx, row in enumerate(matrix):
    #     row[idx] = 1

    # 从标签创建特征矩阵
    features = [[0 for _ in range(6)] for _ in range(9)]
    features[0][5] = 1  # 全局节点
    features[1][3] = 1  # 输入节点
    features[-1][4] = 1  # 输出节点
    for idx, op in enumerate(labels):
        if op is not None and op != 'input' and op != 'output':
            features[idx + 1][int(op) - 2] = 1
    features = features[1:][:-1]
    return matrix, features
