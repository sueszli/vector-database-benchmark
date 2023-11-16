"""
The Computer Language Benchmarks Game
http://benchmarksgame.alioth.debian.org/

regex-dna Python 3 #5 program:
contributed by Dominique Wahli
2to3
modified by Justin Peel

fasta Python 3 #3 program:
modified by Ian Osgood
modified again by Heinrich Acker
modified by Justin Peel
Modified by Christopher Sean Forgeron
"""
import bisect
import re
DEFAULT_INIT_LEN = 100000
DEFAULT_RNG_SEED = 42
ALU = 'GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGGGAGGCCGAGGCGGGCGGATCACCTGAGGTCAGGAGTTCGAGACCAGCCTGGCCAACATGGTGAAACCCCGTCTCTACTAAAAATACAAAAATTAGCCGGGCGTGGTGGCGCGCGCCTGTAATCCCAGCTACTCGGGAGGCTGAGGCAGGAGAATCGCTTGAACCCGGGAGGCGGAGGTTGCAGTGAGCCGAGATCGCGCCACTGCACTCCAGCCTGGGCGACAGAGCGAGACTCCGTCTCAAAAA'
IUB = list(zip('acgtBDHKMNRSVWY', [0.27, 0.12, 0.12, 0.27] + [0.02] * 11))
HOMOSAPIENS = [('a', 0.302954942668), ('c', 0.1979883004921), ('g', 0.1975473066391), ('t', 0.3015094502008)]

def make_cumulative(table):
    if False:
        print('Hello World!')
    P = []
    C = []
    prob = 0.0
    for (char, p) in table:
        prob += p
        P += [prob]
        C += [ord(char)]
    return (P, C)

def repeat_fasta(src, n, nprint):
    if False:
        return 10
    width = 60
    is_trailing_line = False
    count_modifier = 0.0
    len_of_src = len(src)
    ss = src + src + src[:n % len_of_src]
    s = bytearray(ss, encoding='utf8')
    if n % width:
        is_trailing_line = True
        count_modifier = 1.0
    count = 0
    end = n / float(width) - count_modifier
    while count < end:
        i = count * 60 % len_of_src
        nprint(s[i:i + 60] + b'\n')
        count += 1
    if is_trailing_line:
        nprint(s[-(n % width):] + b'\n')

def random_fasta(table, n, seed, nprint):
    if False:
        for i in range(10):
            print('nop')
    width = 60
    r = range(width)
    bb = bisect.bisect
    is_trailing_line = False
    count_modifier = 0.0
    line = bytearray(width + 1)
    (probs, chars) = make_cumulative(table)
    im = 139968.0
    seed = float(seed)
    if n % width:
        is_trailing_line = True
        count_modifier = 1.0
    count = 0.0
    end = n / float(width) - count_modifier
    while count < end:
        for i in r:
            seed = (seed * 3877.0 + 29573.0) % 139968.0
            line[i] = chars[bb(probs, seed / im)]
        line[60] = 10
        nprint(line)
        count += 1.0
    if is_trailing_line:
        for i in range(n % width):
            seed = (seed * 3877.0 + 29573.0) % 139968.0
            line[i] = chars[bb(probs, seed / im)]
        nprint(line[:i + 1] + b'\n')
    return seed

def init_benchmarks(n, rng_seed):
    if False:
        print('Hello World!')
    result = bytearray()
    nprint = result.extend
    nprint(b'>ONE Homo sapiens alu\n')
    repeat_fasta(ALU, n * 2, nprint=nprint)
    nprint(b'>TWO IUB ambiguity codes\n')
    seed = random_fasta(IUB, n * 3, seed=rng_seed, nprint=nprint)
    nprint(b'>THREE Homo sapiens frequency\n')
    random_fasta(HOMOSAPIENS, n * 5, seed, nprint=nprint)
    return bytes(result)
VARIANTS = (b'agggtaaa|tttaccct', b'[cgt]gggtaaa|tttaccc[acg]', b'a[act]ggtaaa|tttacc[agt]t', b'ag[act]gtaaa|tttac[agt]ct', b'agg[act]taaa|ttta[agt]cct', b'aggg[acg]aaa|ttt[cgt]ccct', b'agggt[cgt]aa|tt[acg]accct', b'agggta[cgt]a|t[acg]taccct', b'agggtaa[cgt]|[acg]ttaccct')
SUBST = ((b'B', b'(c|g|t)'), (b'D', b'(a|g|t)'), (b'H', b'(a|c|t)'), (b'K', b'(g|t)'), (b'M', b'(a|c)'), (b'N', b'(a|c|g|t)'), (b'R', b'(a|g)'), (b'S', b'(c|g)'), (b'V', b'(a|c|g)'), (b'W', b'(a|t)'), (b'Y', b'(c|t)'))

def run_benchmarks(seq):
    if False:
        for i in range(10):
            print('nop')
    ilen = len(seq)
    seq = re.sub(b'>.*\n|\n', b'', seq)
    clen = len(seq)
    results = []
    for f in VARIANTS:
        results.append(len(re.findall(f, seq)))
    for (f, r) in SUBST:
        seq = re.sub(f, r, seq)
    return (results, ilen, clen, len(seq))

def bench_regex_dna(loops, seq, expected_res):
    if False:
        return 10
    range_it = range(loops)
    for i in range_it:
        res = run_benchmarks(seq)
    if expected_res is not None and res != expected_res:
        raise Exception('run_benchmarks() error')

def run_benchmark():
    if False:
        print('Hello World!')
    seq = init_benchmarks(DEFAULT_INIT_LEN, DEFAULT_RNG_SEED)
    expected_res = ([6, 26, 86, 58, 113, 31, 31, 32, 43], 1016745, 1000000, 1336326)
    bench_regex_dna(1, seq, expected_res)
if __name__ == '__main__':
    run_benchmark()