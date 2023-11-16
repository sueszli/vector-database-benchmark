"""Same as bench_oneshot.py but uses perf module instead, which is
supposed to be more precise.
"""
import sys
import pyperf
from bench_oneshot import names
import psutil
p = psutil.Process()
funs = [getattr(p, n) for n in names]

def call_normal():
    if False:
        return 10
    for fun in funs:
        fun()

def call_oneshot():
    if False:
        print('Hello World!')
    with p.oneshot():
        for fun in funs:
            fun()

def add_cmdline_args(cmd, args):
    if False:
        print('Hello World!')
    cmd.append(args.benchmark)

def main():
    if False:
        while True:
            i = 10
    runner = pyperf.Runner()
    args = runner.parse_args()
    if not args.worker:
        print('%s methods involved on platform %r (psutil %s):' % (len(names), sys.platform, psutil.__version__))
        for name in sorted(names):
            print('    ' + name)
    runner.bench_func('normal', call_normal)
    runner.bench_func('oneshot', call_oneshot)
main()