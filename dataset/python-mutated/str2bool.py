import argparse

def str2bool(val):
    if False:
        while True:
            i = 10
    '\n    Resolving boolean arguments if they are not given in the standard format\n\n    Arguments:\n        val (bool or string): boolean argument type\n    Output:\n        bool: the desired value {True, False}\n\n    '
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        if val.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif val.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')