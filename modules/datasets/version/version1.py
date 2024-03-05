# coding : utf-8
# Author : yuxiang Zeng
import pickle
import torch
from utils.utils import *


class experiment1:
    def __init__(self, args):
        self.args = args

    def load_data(self, args):
        import os
        # 获取目录下的所有文件名
        file_names = os.listdir(args.path)
        # print(file_names)
        # 筛选出只有.pickle扩展名的文件
        pickle_files = [file for file in file_names if file.endswith('.pickle')]
        # 打印.pickle文件名
        pickle_file = args.path + pickle_files[int(args.dataset)]
        # print(pickle_file)
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        # print(type(data))
        return data

    def preprocess_data(self, data, args):
        tensor = []
        for key in data.keys():
            now = []
            for x in key:
                now.append(x)
            y = data[key]
            now.append(y)
            tensor.append(now)
        tensor = np.array(tensor)
        return tensor

    def get_pytorch_index(self, data):
        return torch.as_tensor(data)
