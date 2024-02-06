# coding : utf-8
# Author : yuxiang Zeng
import pickle
import torch
from utils.utils import *


class experiment3:
    def __init__(self, args):
        self.args = args

    def load_data(self, args):
        import os
        file_names = os.listdir(args.path)
        pickle_files = [file for file in file_names if file.endswith('.pickle')]
        data = []
        for i in range(len(pickle_files)):
            pickle_file = args.path + pickle_files[i]
            # print(pickle_file)
            with open(pickle_file, 'rb') as f:
                now = pickle.load(f)
            data.append(now)
        return data

    def preprocess_data(self, data, args):
        tensor = []
        for i in range(len(data)):
            for key in data[i].keys():
                now = []
                now.append(i)
                for x in key:
                    now.append(x)
                y = data[i][key]
                now.append(y)
                tensor.append(now)
        tensor = np.array(tensor)
        return tensor

    def get_pytorch_index(self, data):
        return torch.as_tensor(data)
