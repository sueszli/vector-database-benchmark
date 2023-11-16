import argparse
import yaml
from parse_utils import parse_op_entry

def main(op_yaml_path, output_path, backward):
    if False:
        print('Hello World!')
    with open(op_yaml_path, 'rt') as f:
        ops = yaml.safe_load(f)
        if ops is None:
            ops = []
        else:
            ops = [parse_op_entry(op, 'backward_op' if backward else 'op') for op in ops]
    with open(output_path, 'wt') as f:
        yaml.safe_dump(ops, f, default_flow_style=None, sort_keys=False)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse op yaml into canonical format.')
    parser.add_argument('--op_yaml_path', type=str, help='op yaml file.')
    parser.add_argument('--output_path', type=str, help='path to save parsed yaml file.')
    parser.add_argument('--backward', action='store_true', default=False)
    args = parser.parse_args()
    main(args.op_yaml_path, args.output_path, args.backward)