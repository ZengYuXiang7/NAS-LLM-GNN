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
    for exper in [5]:
        # for device_type in ['cpu', 'gpu', 'tpu', 'dsp']:
        for device_type in ['cpu']:
            for dim in [128]:
                for density in [0.10]:
                    if exper in [4, 5]:
                        args.path = './datasets/' + device_type + '/'
                    # args.debug = 1
                    args.rounds = 5
                    args.exper = exper
                    args.model = str(exper)
                    args.density = density
                    args.epochs = 150
                    args.bs = 32
                    args.lr = args.decay = 0.001
                    args.dimension = dim
                    args.experiment = 1
                    args.valid = 1
                    set_settings(args)
                    log = Logger(args)
                    log(str(args))
                    log(f'{device_type}_Density_{density:.2f}')
                    args.log = log
                    RunExperiments(log, args)


if __name__ == '__main__':
    args = get_args()
    run_in_windows()

