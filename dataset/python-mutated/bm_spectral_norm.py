"""
MathWorld: "Hundred-Dollar, Hundred-Digit Challenge Problems", Challenge #3.
http://mathworld.wolfram.com/Hundred-DollarHundred-DigitChallengeProblems.html

The Computer Language Benchmarks Game
http://benchmarksgame.alioth.debian.org/u64q/spectralnorm-description.html#spectralnorm

Contributed by Sebastien Loisel
Fixed by Isaac Gouy
Sped up by Josh Goldfoot
Dirtily sped up by Simon Descarpentries
Concurrency by Jason Stitt
"""
from six.moves import xrange, zip as izip
DEFAULT_N = 130

def eval_A(i, j):
    if False:
        while True:
            i = 10
    return 1.0 / ((i + j) * (i + j + 1) // 2 + i + 1)

def eval_times_u(func, u):
    if False:
        return 10
    return [func((i, u)) for i in xrange(len(list(u)))]

def eval_AtA_times_u(u):
    if False:
        i = 10
        return i + 15
    return eval_times_u(part_At_times_u, eval_times_u(part_A_times_u, u))

def part_A_times_u(i_u):
    if False:
        i = 10
        return i + 15
    (i, u) = i_u
    partial_sum = 0
    for (j, u_j) in enumerate(u):
        partial_sum += eval_A(i, j) * u_j
    return partial_sum

def part_At_times_u(i_u):
    if False:
        for i in range(10):
            print('nop')
    (i, u) = i_u
    partial_sum = 0
    for (j, u_j) in enumerate(u):
        partial_sum += eval_A(j, i) * u_j
    return partial_sum

def bench_spectral_norm(loops):
    if False:
        i = 10
        return i + 15
    range_it = xrange(loops)
    for _ in range_it:
        u = [1] * DEFAULT_N
        for dummy in xrange(10):
            v = eval_AtA_times_u(u)
            u = eval_AtA_times_u(v)
        vBv = vv = 0
        for (ue, ve) in izip(u, v):
            vBv += ue * ve
            vv += ve * ve
    return
if __name__ == '__main__':
    bench_spectral_norm(10)