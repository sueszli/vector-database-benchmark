"""
the code for
jupytext-config set-default-viewer
and related subcommands
"""
import sys
from argparse import ArgumentParser
from .labconfig import LabConfig

class SubCommand:
    """
    a subcommand for jupytext-config
    """

    def __init__(self, name, help):
        if False:
            print('Hello World!')
        self.name = name
        self.help = help

    def main(self, args):
        if False:
            i = 10
            return i + 15
        '\n        return 0 if all goes well\n        '
        print(f'{self.__class__.__name__}: redefine main() to implement this subcommand')
        return 1

class ListDefaultViewer(SubCommand):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__('list-default-viewer', 'Display current settings in labconfig/')

    def main(self, args):
        if False:
            print('Hello World!')
        LabConfig().read().list_default_viewer()
        return 0

    def fill_parser(self, subparser):
        if False:
            for i in range(10):
                print('nop')
        pass

class SetDefaultViewer(SubCommand):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__('set-default-viewer', 'Set default viewers for JupyterLab')

    def main(self, args):
        if False:
            print('Hello World!')
        LabConfig().read().set_default_viewers(args.doctype).write()
        return 0

    def fill_parser(self, subparser):
        if False:
            return 10
        subparser.add_argument('doctype', nargs='*', help=f"the document types to be associated with the notebook editor; defaults to {' '.join(LabConfig.DOCTYPES)}")

class UnsetDefaultViewer(SubCommand):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__('unset-default-viewer', 'Unset default viewers for JupyterLab')

    def main(self, args):
        if False:
            return 10
        LabConfig().read().unset_default_viewers(args.doctype).write()
        return 0

    def fill_parser(self, subparser):
        if False:
            while True:
                i = 10
        subparser.add_argument('doctype', nargs='*', help=f"the document types for which the default viewer will be unset; defaults to {' '.join(LabConfig.DOCTYPES)}")
SUBCOMMANDS = [ListDefaultViewer(), SetDefaultViewer(), UnsetDefaultViewer()]

def main():
    if False:
        print('Hello World!')
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    for subcommand in SUBCOMMANDS:
        subparser = subparsers.add_parser(subcommand.name, help=subcommand.help)
        subparser.set_defaults(subcommand=subcommand)
        subcommand.fill_parser(subparser)
    args = parser.parse_args(sys.argv[1:] or ['--help'])
    return args.subcommand.main(args)