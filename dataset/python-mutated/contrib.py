"""Module for handling contributed support."""
import argparse
from ludwig.contribs import contrib_registry, ContribLoader

def create_load_action(contrib_loader: ContribLoader) -> argparse.Action:
    if False:
        return 10

    class LoadContribAction(argparse.Action):

        def __call__(self, parser, namespace, values, option_string):
            if False:
                for i in range(10):
                    print('nop')
            items = getattr(namespace, self.dest) or []
            items.append(contrib_loader.load())
            setattr(namespace, self.dest, items)
    return LoadContribAction

def add_contrib_callback_args(parser: argparse.ArgumentParser):
    if False:
        for i in range(10):
            print('nop')
    for (contrib_name, contrib_loader) in contrib_registry.items():
        parser.add_argument(f'--{contrib_name}', dest='callbacks', nargs=0, action=create_load_action(contrib_loader))

def preload(argv):
    if False:
        i = 10
        return i + 15
    for arg in argv:
        if arg.startswith('--'):
            arg = arg[2:]
        if arg in contrib_registry:
            contrib_registry[arg].preload()