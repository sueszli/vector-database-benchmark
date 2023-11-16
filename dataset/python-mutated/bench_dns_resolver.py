from __future__ import absolute_import, print_function, division
from gevent import monkey
monkey.patch_all()
import sys
import socket
import perf
import gevent
from zope.dottedname.resolve import resolve as drresolve
blacklist = {22, 55, 68, 69, 72, 52, 94, 62, 54, 71, 73, 74, 34, 36, 83, 86, 79, 81, 98, 99, 120, 130, 152, 161, 165, 169, 172, 199, 205, 239, 235, 254, 256, 286, 299, 259, 229, 190, 185, 182, 173, 160, 158, 153, 139, 138, 131, 129, 127, 125, 116, 112, 110, 106}
RUN_COUNT = 15 if hasattr(sys, 'pypy_version_info') else 5

def quiet(f, n):
    if False:
        i = 10
        return i + 15
    try:
        f(n)
    except socket.gaierror:
        pass

def resolve_seq(res, count=10, begin=0):
    if False:
        while True:
            i = 10
    for index in range(begin, count + begin):
        if index in blacklist:
            continue
        try:
            res.gethostbyname('x%s.com' % index)
        except socket.gaierror:
            pass

def resolve_par(res, count=10, begin=0):
    if False:
        for i in range(10):
            print('nop')
    gs = []
    for index in range(begin, count + begin):
        if index in blacklist:
            continue
        gs.append(gevent.spawn(quiet, res.gethostbyname, 'x%s.com' % index))
    gevent.joinall(gs)
N = 300

def run_all(resolver_name, resolve):
    if False:
        print('Hello World!')
    res = drresolve('gevent.resolver.' + resolver_name + '.Resolver')
    res = res()
    res._getaliases = lambda hostname, family: []
    if N > 150:
        count = N // 3
        resolve(res, count=count)
        resolve(res, count=count, begin=count)
        resolve(res, count=count, begin=count * 2)
    else:
        resolve(res, count=N)

def main():
    if False:
        return 10

    def worker_cmd(cmd, args):
        if False:
            while True:
                i = 10
        cmd.extend(args.benchmark)
    runner = perf.Runner(processes=5, values=3, add_cmdline_args=worker_cmd)
    all_names = ('dnspython', 'blocking', 'ares', 'thread')
    runner.argparser.add_argument('benchmark', nargs='*', default='all', choices=all_names + ('all',))
    args = runner.parse_args()
    if 'all' in args.benchmark or args.benchmark == 'all':
        args.benchmark = ['all']
        names = all_names
    else:
        names = args.benchmark
    for name in names:
        runner.bench_func(name + ' sequential', run_all, name, resolve_seq, inner_loops=N)
        runner.bench_func(name + ' parallel', run_all, name, resolve_par, inner_loops=N)
if __name__ == '__main__':
    main()