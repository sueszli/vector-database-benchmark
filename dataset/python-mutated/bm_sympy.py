import pyperf
from sympy import expand, symbols, integrate, tan, summation
from sympy.core.cache import clear_cache

def bench_expand():
    if False:
        i = 10
        return i + 15
    (x, y, z) = symbols('x y z')
    return expand((1 + x + y + z) ** 20)

def bench_integrate():
    if False:
        i = 10
        return i + 15
    (x, y) = symbols('x y')
    f = (1 / tan(x)) ** 10
    return integrate(f, x)

def bench_sum():
    if False:
        return 10
    (x, i) = symbols('x i')
    summation(x ** i / i, (i, 1, 400))

def bench_str():
    if False:
        return 10
    (x, y, z) = symbols('x y z')
    str(expand((x + 2 * y + 3 * z) ** 30))

def bench_sympy(loops, func):
    if False:
        i = 10
        return i + 15
    timer = pyperf.perf_counter
    dt = 0
    for _ in range(loops):
        clear_cache()
        t0 = timer()
        func()
        dt += timer() - t0
    return dt
BENCHMARKS = ('expand', 'integrate', 'sum', 'str')

def add_cmdline_args(cmd, args):
    if False:
        print('Hello World!')
    if args.benchmark:
        cmd.append(args.benchmark)
if __name__ == '__main__':
    runner = pyperf.Runner(add_cmdline_args=add_cmdline_args)
    runner.metadata['description'] = 'SymPy benchmark'
    runner.argparser.add_argument('benchmark', nargs='?', choices=BENCHMARKS)
    import gc
    gc.disable()
    args = runner.parse_args()
    if args.benchmark:
        benchmarks = (args.benchmark,)
    else:
        benchmarks = BENCHMARKS
    for bench in benchmarks:
        name = 'sympy_%s' % bench
        func = globals()['bench_' + bench]
        func()