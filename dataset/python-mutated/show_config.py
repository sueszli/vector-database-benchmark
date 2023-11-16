from __future__ import print_function
import yaml

def main(args):
    if False:
        i = 10
        return i + 15
    with open(args.config) as fle:
        config = yaml.safe_load(fle)
    print(yaml.dump(config, default_flow_style=False))