# coding : utf-8
# Author : yuxiang Zeng

from modules.main import RunExperiments, get_args
from utils.logger import Logger
from utils.utils import set_settings


# Run Experiments
def run_in_windows():
    # 1 只有模型层， 但是单计算结点
    # 2 序列化
    # 3 低秩处理  含计算节点embed
    # 4 低秩处理  含简单设备序列号
    # 5 大语言模型
    # 6 大语言模型 + GNN
    # for exper in [5]:
    for exper in [6]:
        for device_type in ['cpu']:
            for dim in [128]:
                for density in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    if exper in [4, 5, 6]:
                        args.path = './datasets/' + device_type + '/'
                    args.rounds = 5
                    args.exper = exper
                    args.model = str(exper)
                    args.density = density
                    args.epochs = 150
                    args.bs = 32
                    args.dimension = dim
                    args.experiment = 1
                    # 慢设备运行
                    args.verbose = 1
                    args.program_test = 1
                    set_settings(args)
                    log = Logger(args)
                    log(str(args))
                    log(f'{device_type}_Density_{density:.2f}')
                    args.log = log
                    RunExperiments(log, args)


if __name__ == '__main__':
    args = get_args()
    run_in_windows()

