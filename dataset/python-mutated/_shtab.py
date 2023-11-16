FILE = None
DIRECTORY = DIR = None

def add_argument_to(parser, *args, **kwargs):
    if False:
        return 10
    from argparse import Action
    Action.complete = None
    return parser