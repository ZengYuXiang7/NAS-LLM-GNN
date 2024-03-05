# coding : utf-8
# Author : yuxiang Zeng
import pickle
import torch
from utils.utils import *


class experiment4:
    def __init__(self, args):
        self.args = args

    def get_device_item(self, file_names):
        import re, pandas
        from sklearn.preprocessing import LabelEncoder
        # 定义正则表达式模式
        pattern = re.compile(r'^(?P<device_type>\w+)-(?P<unit>\w+)-(?P<devices>[\w-]+)-(?P<precision>\w+).pickle$')
        # 解析文件名并存储在列表中
        data = []
        for file_name in file_names:
            match = pattern.match(file_name)
            if match:
                data.append(match.groupdict())
        df = pandas.DataFrame(data).to_numpy()
        # print(df)
        # 对每一列进行标签编码
        label_encoder = LabelEncoder()
        encoded_data = np.apply_along_axis(label_encoder.fit_transform, axis=0, arr=df)
        # print(encoded_data)
        return encoded_data

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
        # print(type(device_label), type(data))
        # print(device_label.shape, data.shape)
        data = np.concatenate([device_label, data], axis=1)
        # print(data.shape)
        data = list(data)
        # print(len(data))
        return data

    def preprocess_data(self, data, args):
        # print("数据预处理")
        tensor = []
        for i in range(len(data)):
            for key in data[i][4].keys():
                now = []
                # 添加设备号
                for j in range(4):
                    now.append(data[i][j])
                # 添加模型号
                for x in key:
                    now.append(x)
                y = data[i][4][key]
                now.append(y)
                tensor.append(now)
                # print(now)
        # print(tensor)
        tensor = np.array(tensor)
        return tensor

    def get_pytorch_index(self, data):
        return torch.as_tensor(data)
