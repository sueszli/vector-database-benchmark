"""Detect reference leaks after several unit test runs.

The script runs the unit test and counts the objects alive after the run. If
the object count differs between the last two runs, a report is printed and the
script exits with error 1.
"""
import argparse
import gc
import sys
import difflib
import unittest
from pprint import pprint
from collections import defaultdict

def main():
    if False:
        for i in range(10):
            print('nop')
    opt = parse_args()
    import tests
    test = tests
    if opt.suite:
        test = getattr(test, opt.suite)
    sys.stdout.write(f'test suite {test.__name__}\n')
    for i in range(1, opt.nruns + 1):
        sys.stdout.write(f'test suite run {i} of {opt.nruns}\n')
        runner = unittest.TextTestRunner()
        runner.run(test.test_suite())
        dump(i, opt)
    f1 = open(f'debug-{opt.nruns - 1:02}.txt').readlines()
    f2 = open(f'debug-{opt.nruns:02}.txt').readlines()
    for line in difflib.unified_diff(f1, f2, f'run {opt.nruns - 1}', f'run {opt.nruns}'):
        sys.stdout.write(line)
    rv = f1 != f2 and 1 or 0
    if opt.objs:
        f1 = open(f'objs-{opt.nruns - 1:02}.txt').readlines()
        f2 = open(f'objs-{opt.nruns:02}.txt').readlines()
        for line in difflib.unified_diff(f1, f2, f'run {opt.nruns - 1}', f'run {opt.nruns}'):
            sys.stdout.write(line)
    return rv

def parse_args():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nruns', type=int, metavar='N', default=3, help='number of test suite runs [default: %(default)d]')
    parser.add_argument('--suite', metavar='NAME', help="the test suite to run (e.g. 'test_cursor'). [default: all]")
    parser.add_argument('--objs', metavar='TYPE', help='in case of leaks, print a report of object TYPE (support still incomplete)')
    return parser.parse_args()

def dump(i, opt):
    if False:
        return 10
    gc.collect()
    objs = gc.get_objects()
    c = defaultdict(int)
    for o in objs:
        c[type(o)] += 1
    pprint(sorted(((v, str(k)) for (k, v) in c.items()), reverse=True), stream=open(f'debug-{i:02}.txt', 'w'))
    if opt.objs:
        co = []
        t = getattr(__builtins__, opt.objs)
        for o in objs:
            if type(o) is t:
                co.append(o)
        if t is dict:
            co.sort(key=lambda d: d.items())
        else:
            co.sort()
        pprint(co, stream=open(f'objs-{i:02}.txt', 'w'))
if __name__ == '__main__':
    sys.exit(main())