import argparse
from tensorflow.contrib.framework.python.framework import checkpoint_utils

def get_arg_parser():
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', help='Path to the model checkpoint.', type=str, required=False)
    return parser
parser = get_arg_parser()
args = parser.parse_args()
checkpoint_dir = args.checkpoint
for (name, shape) in checkpoint_utils.list_variables(checkpoint_dir):
    if 'embedding' in name:
        print(name, shape, checkpoint_utils.load_variable(checkpoint_dir, name))