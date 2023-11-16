"""Main program for testing the infrastructure."""
from __future__ import print_function
__author__ = 'Guido van Rossum <guido@python.org>'
from . import support
import os
import sys
import logging
from .. import pytree
from .. import pgen2
from ..pgen2 import driver
logging.basicConfig()

def main():
    if False:
        print('Hello World!')
    gr = driver.load_grammar('Grammar.txt')
    dr = driver.Driver(gr, convert=pytree.convert)
    fn = 'example.py'
    tree = dr.parse_file(fn, debug=True)
    if not diff(fn, tree):
        print('No diffs.')
    if not sys.argv[1:]:
        return
    problems = []
    for name in sys.modules:
        mod = sys.modules[name]
        if mod is None or not hasattr(mod, '__file__'):
            continue
        fn = mod.__file__
        if fn.endswith('.pyc'):
            fn = fn[:-1]
        if not fn.endswith('.py'):
            continue
        print('Parsing', fn, file=sys.stderr)
        tree = dr.parse_file(fn, debug=True)
        if diff(fn, tree):
            problems.append(fn)
    for dir in sys.path:
        try:
            names = os.listdir(dir)
        except OSError:
            continue
        print('Scanning', dir, '...', file=sys.stderr)
        for name in names:
            if not name.endswith('.py'):
                continue
            print('Parsing', name, file=sys.stderr)
            fn = os.path.join(dir, name)
            try:
                tree = dr.parse_file(fn, debug=True)
            except pgen2.parse.ParseError as err:
                print('ParseError:', err)
            else:
                if diff(fn, tree):
                    problems.append(fn)
    if not problems:
        print('No problems.  Congratulations!')
    else:
        print('Problems in following files:')
        for fn in problems:
            print('***', fn)

def diff(fn, tree):
    if False:
        return 10
    f = open('@', 'w')
    try:
        f.write(str(tree))
    finally:
        f.close()
    try:
        return os.system('diff -u %s @' % fn)
    finally:
        os.remove('@')
if __name__ == '__main__':
    main()