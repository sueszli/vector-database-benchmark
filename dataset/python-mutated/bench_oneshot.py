"""A simple micro benchmark script which prints the speedup when using
Process.oneshot() ctx manager.
See: https://github.com/giampaolo/psutil/issues/799.
"""
from __future__ import division
from __future__ import print_function
import sys
import textwrap
import timeit
import psutil
ITERATIONS = 1000
names = ['cpu_times', 'cpu_percent', 'memory_info', 'memory_percent', 'ppid', 'parent']
if psutil.POSIX:
    names.append('uids')
    names.append('username')
if psutil.LINUX:
    names += ['cpu_num', 'cpu_times', 'gids', 'name', 'num_ctx_switches', 'num_threads', 'ppid', 'status', 'terminal', 'uids']
elif psutil.BSD:
    names = ['cpu_times', 'gids', 'io_counters', 'memory_full_info', 'memory_info', 'name', 'num_ctx_switches', 'ppid', 'status', 'terminal', 'uids']
    if psutil.FREEBSD:
        names.append('cpu_num')
elif psutil.SUNOS:
    names += ['cmdline', 'gids', 'memory_full_info', 'memory_info', 'name', 'num_threads', 'ppid', 'status', 'terminal', 'uids']
elif psutil.MACOS:
    names += ['cpu_times', 'create_time', 'gids', 'memory_info', 'name', 'num_ctx_switches', 'num_threads', 'ppid', 'terminal', 'uids']
elif psutil.WINDOWS:
    names += ['num_ctx_switches', 'num_threads', 'num_handles', 'cpu_times', 'create_time', 'num_threads', 'io_counters', 'memory_info']
names = sorted(set(names))
setup = textwrap.dedent('\n    from __main__ import names\n    import psutil\n\n    def call_normal(funs):\n        for fun in funs:\n            fun()\n\n    def call_oneshot(funs):\n        with p.oneshot():\n            for fun in funs:\n                fun()\n\n    p = psutil.Process()\n    funs = [getattr(p, n) for n in names]\n    ')

def main():
    if False:
        while True:
            i = 10
    print('%s methods involved on platform %r (%s iterations, psutil %s):' % (len(names), sys.platform, ITERATIONS, psutil.__version__))
    for name in sorted(names):
        print('    ' + name)
    elapsed1 = timeit.timeit('call_normal(funs)', setup=setup, number=ITERATIONS)
    print('normal:  %.3f secs' % elapsed1)
    elapsed2 = timeit.timeit('call_oneshot(funs)', setup=setup, number=ITERATIONS)
    print('onshot:  %.3f secs' % elapsed2)
    if elapsed2 < elapsed1:
        print('speedup: +%.2fx' % (elapsed1 / elapsed2))
    elif elapsed2 > elapsed1:
        print('slowdown: -%.2fx' % (elapsed2 / elapsed1))
    else:
        print('same speed')
if __name__ == '__main__':
    main()