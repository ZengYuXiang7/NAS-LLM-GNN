# coding : utf-8
# Author : yuxiang Zeng

from modules.main import RunExperiments, get_args
from utils.logger import Logger
from utils.utils import set_settings


# Run Experiments
def run_in_py():
    # 1 只有模型层， 但是单计算结点
    # 2 序列化
    # 3 低秩处理  含计算节点embed
    # 4 低秩处理  含简单设备序列号
    # 5 大语言模型
    # 6 大语言模型 + GNN
    # 7 大模型 + GNN修改 + 嵌入重构 + GNN强化
    def Runonce(args):
        set_settings(args)
        log = Logger(args)
        log.log(str(args))
        log.log(f'{device_type}_Density_{args.density:.2f}')
        args.log = log
        RunExperiments(log, args)

    for exper in [7]:
        for device_type in ['cpu']:
            for dim in [128]:
                for density in [0.1]:
                    if exper in [4, 5, 6, 7]:
                        args.path = './datasets/' + device_type + '/'
                    args.rounds = 5
                    args.exper = exper
                    args.model = str(exper)
                    args.density = density
                    args.epochs = 300
                    args.bs = 32
                    args.dimension = dim
                    # 慢设备运行
                    args.verbose = 1
                    args.program_test = 0
                    Runonce(args)


if __name__ == '__main__':
    args = get_args()
    run_in_py()

