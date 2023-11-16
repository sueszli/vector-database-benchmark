from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import itertools
import math
import operator
import sys
if sys.version_info.major == 2:
    map = itertools.imap

def average(x):
    if False:
        i = 10
        return i + 15
    return math.fsum(x) / len(x)

def variance(x):
    if False:
        while True:
            i = 10
    avgx = average(x)
    return list(map(lambda y: (y - avgx) ** 2, x))

def standarddev(x):
    if False:
        while True:
            i = 10
    return math.sqrt(average(variance(x)))

def run(pargs=None):
    if False:
        for i in range(10):
            print('nop')
    args = parse_args(pargs)
    returns = [args.ret1, args.ret2]
    retfree = args.riskfreerate
    print('returns is:', returns, ' - retfree is:', retfree)
    retfree = itertools.repeat(retfree)
    ret_free = map(operator.sub, returns, retfree)
    ret_free_avg = average(list(ret_free))
    print('returns excess mean:', ret_free_avg)
    retdev = standarddev(returns)
    print('returns standard deviation:', retdev)
    ratio = ret_free_avg / retdev
    print('Sharpe Ratio is:', ratio)

def parse_args(pargs=None):
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Sample Sharpe Ratio')
    parser.add_argument('--ret1', required=False, action='store', type=float, default=0.023286, help='Annual Return 1')
    parser.add_argument('--ret2', required=False, action='store', type=float, default=0.0257816485323, help='Annual Return 2')
    parser.add_argument('--riskfreerate', required=False, action='store', type=float, default=0.01, help='Risk free rate (decimal) for the Sharpe Ratio')
    if pargs is not None:
        return parser.parse_args(pargs)
    return parser.parse_args()
if __name__ == '__main__':
    run()