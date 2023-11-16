import argparse
from itertools import chain
from pathlib import Path
import yaml
from parse_utils import cross_validate, to_named_dict

def main(forward_op_yaml_paths, backward_op_yaml_paths):
    if False:
        for i in range(10):
            print('nop')
    ops = {}
    for op_yaml_path in chain(forward_op_yaml_paths, backward_op_yaml_paths):
        with open(op_yaml_path, 'rt', encoding='utf-8') as f:
            op_list = yaml.safe_load(f)
            if op_list is not None:
                ops.update(to_named_dict(op_list))
    cross_validate(ops)
if __name__ == '__main__':
    current_dir = Path(__file__).parent / 'temp'
    parser = argparse.ArgumentParser(description='Parse op yaml into canonical format.')
    parser.add_argument('--forward_yaml_paths', type=str, nargs='+', default=[str(current_dir / 'op .parsed.yaml')], help='forward op yaml file.')
    parser.add_argument('--backward_yaml_paths', type=str, nargs='+', default=[str(current_dir / 'backward_op .parsed.yaml')], help='backward op yaml file.')
    args = parser.parse_args()
    main(args.forward_yaml_paths, args.backward_yaml_paths)