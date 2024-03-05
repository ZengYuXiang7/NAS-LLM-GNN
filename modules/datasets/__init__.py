from modules.datasets.version.version1 import experiment1
from modules.datasets.version.version2 import experiment2
from modules.datasets.version.version3 import experiment3
from modules.datasets.version.version4 import experiment4
from modules.datasets.version.version5 import experiment5
from modules.datasets.version.version6 import experiment6
from modules.datasets.version.version7 import experiment7
from modules.datasets.version.version8 import experiment8
from modules.datasets.version.version9 import experiment9


def get_exper(args):
    if args.exper == 1:
        return experiment1(args)
    elif args.exper == 2:
        return experiment2(args)
    elif args.exper == 3:
        return experiment3(args)
    elif args.exper == 4:
        return experiment4(args)
    elif args.exper == 5:
        return experiment5(args)
    elif args.exper == 6:
        return experiment6(args)
    elif args.exper == 7:
        return experiment7(args)
    elif args.exper == 8:
        return experiment8(args)
    elif args.exper == 9:
        return experiment9(args)
