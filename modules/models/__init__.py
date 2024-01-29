from modules.models.version.version1 import NAS_Model
from modules.models.version.version2 import NAS_Model_2
from modules.models.version.version3 import NAS_Model_3
from modules.models.version.version4 import NAS_Model_4
from modules.models.version.version5 import NAS_Model_Chatgpt
from modules.models.version.version6 import NAS_Model_Chatgpt_GNN
from modules.models.version.version7 import NAS_Model_Chatgpt_GNN_2
from modules.models.version.version8 import NAS_Model_Chatgpt_GNN_3


def get_model(args):
    if args.model == '1':
        return NAS_Model(args)
    elif args.model == '2':
        return NAS_Model_2(args)  # Only计算节点Embedding
    elif args.model == '3':
        return NAS_Model_3(args)  # 带设备序列号
    elif args.model == '4':
        return NAS_Model_4(args)  # 带设备序列号
    elif args.model == '5':
        return NAS_Model_Chatgpt(args)  # 大语言模型
    elif args.model == '6':
        return NAS_Model_Chatgpt_GNN(args)  # 大语言模型 + GNN
    elif args.model == '7':
        return NAS_Model_Chatgpt_GNN_2(args)  # 大语言模型 + GNN
    elif args.model == '8':
        return NAS_Model_Chatgpt_GNN_3(args)  # 大语言模型 + GNN
    else:
        raise NotImplementedError
