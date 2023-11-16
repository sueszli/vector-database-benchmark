import argparse
from allennlp.commands import Subcommand

def do_nothing(_):
    if False:
        return 10
    pass

@Subcommand.register('d')
class D(Subcommand):

    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        if False:
            for i in range(10):
                print('nop')
        subparser = parser.add_parser(self.name, description='fake', help='fake help')
        subparser.set_defaults(func=do_nothing)
        return subparser