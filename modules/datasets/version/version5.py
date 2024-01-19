# coding : utf-8
# Author : yuxiang Zeng
import pickle
import torch
from utils.utils import *


class experiment5:
    def __init__(self, args):
        self.args = args
        self.chatgpt = NAS_ChatGPT(args)

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
        # for i in range(len(df)):
        #     print(f'Device {df[i, 2]} Info :', self.chatgpt.get_device_more_info(df[i, 2]))

        # 大语言模型进行数据增强
        try:
            makedir('./pretrained')
            with open(f'./pretrained/pretrained.pkl', 'rb') as f:
                device_info = pickle.load(f)
        except:
            print(df)
            device_info = []
            for i in range(len(df)):
                device_base_info = df[i, 2]
                device_more_info = self.chatgpt.get_device_more_info(device_base_info)
                print(device_more_info)
                pattern = re.compile(
                    r'^(?P<frequency>[\d.]+)\sGHz::(?P<cores>\d+)\scores::(?P<threads>\d+)\sThreads::(?P<memory_size>[\d.]+)\sMB::(?P<memory_speed>[\d.]+)\sGB/s$')
                match = pattern.match(device_more_info)
                if match:
                    frequency = float(match.group('frequency'))
                    cores = int(match.group('cores'))
                    threads = int(match.group('threads'))
                    memory_size = float(match.group('memory_size'))
                    memory_speed = float(match.group('memory_speed'))
                device_info.append([frequency, cores, threads, memory_size, memory_speed])
            device_info = np.array(device_info)
            with open(f'./pretrained/pretrained.pkl', 'wb') as f:
                pickle.dump(device_info, f)
        # print(df)                                                  # 打印原始信息
        # print(device_info)                                         # 打印数据增强信息
        device_info = device_info / np.max(device_info, axis=0)  # 归一化
        # device_info = (device_info - np.mean(device_info, axis=0)) / np.std(device_info, axis=0)    # 归一化
        # print(device_info)
        df = np.delete(df, 2, axis=1)  # 删掉设备名
        # print(df)                                                # 打印删掉设备名的信息
        # 对每一列进行标签编码
        label_encoder = LabelEncoder()
        encoded_data = np.apply_along_axis(label_encoder.fit_transform, axis=0, arr=df)
        final_data = np.concatenate([encoded_data, device_info], axis=1)
        # print(encoded_data)
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
        # print('-' * 100)
        # print(data[:, :7])  # 数列在8
        # data = list(data)
        return data

    def preprocess_data(self, data, args):
        tensor = []
        for i in range(len(data)):
            for key in data[i][8].keys():
                now = []
                # 添加设备号
                for j in range(len(data[0]) - 1):
                    now.append(data[i][j])
                # 添加模型号
                for x in key:
                    now.append(x)
                y = data[i][8][key]
                now.append(y)
                tensor.append(now)
                # print(now)
        tensor = np.array(tensor)
        return tensor

    def get_pytorch_index(self, data):
        return torch.as_tensor(data)
