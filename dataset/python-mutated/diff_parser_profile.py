"""
Profile a piece of Python code with ``cProfile`` that uses the diff parser.

Usage:
  profile.py <file> [-d] [-s <sort>]
  profile.py -h | --help

Options:
  -h --help     Show this screen.
  -d --debug    Enable Jedi internal debugging.
  -s <sort>     Sort the profile results, e.g. cumtime, name [default: time].
"""
import cProfile
from docopt import docopt
from jedi.parser.python import load_grammar
from jedi.parser.diff import DiffParser
from jedi.parser.python import ParserWithRecovery
from jedi.common import splitlines
import jedi

def run(parser, lines):
    if False:
        return 10
    diff_parser = DiffParser(parser)
    diff_parser.update(lines)
    parser.module.used_names

def main(args):
    if False:
        for i in range(10):
            print('nop')
    if args['--debug']:
        jedi.set_debug_function(notices=True)
    with open(args['<file>']) as f:
        code = f.read()
    grammar = load_grammar()
    parser = ParserWithRecovery(grammar, code)
    parser.module.used_names
    code = code + '\na\n'
    lines = splitlines(code, keepends=True)
    cProfile.runctx('run(parser, lines)', globals(), locals(), sort=args['-s'])
if __name__ == '__main__':
    args = docopt(__doc__)
    main(args)