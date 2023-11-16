import argparse
from mmcv import Config, DictAction

def parse_args():
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser(description='Print the whole config')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--options', nargs='+', action=DictAction, help='arguments in dict')
    args = parser.parse_args()
    return args

def main():
    if False:
        i = 10
        return i + 15
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    print(f'Config:\n{cfg.pretty_text}')
if __name__ == '__main__':
    main()