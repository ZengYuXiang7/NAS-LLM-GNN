# coding : utf-8
# Author : yuxiang Zeng

from modules.main import RunExperiments, get_args
from utils.logger import Logger
from utils.utils import set_settings

# Run Experiments
def run_in_windows():
    for device_type in ['cpu', 'gpu', 'dsp', 'tpu']:
        for dim in [128]:
            for model in ['3']:
                for density in [0.10]:
                    args.path = './datasets/' + device_type + '/'
                    args.rounds = 3
                    args.exper = 3
                    args.model = model
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
                    args.log = log
                    RunExperiments(log, args)


if __name__ == '__main__':
    args = get_args()
    run_in_windows()

