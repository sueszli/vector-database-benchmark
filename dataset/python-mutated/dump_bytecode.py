import argparse
import dis
import types
from typing import Dict, List, Tuple
CountAssocs = List[Tuple[str, int]]
Histogram = Dict[str, int]

def count_bytecode(func: types.FunctionType, histo: Histogram) -> None:
    if False:
        return 10
    'Return a distribution of bytecode in func'
    for instr in dis.get_instructions(func):
        if instr.opname in histo:
            histo[instr.opname] += 1
        else:
            histo[instr.opname] = 1

def summarize(obj: object, histo: Histogram) -> None:
    if False:
        i = 10
        return i + 15
    'Compute the bytecode histogram of all functions reachable from obj'
    if isinstance(obj, types.FunctionType):
        count_bytecode(obj, histo)
    elif isinstance(obj, type):
        for child in obj.__dict__.values():
            summarize(child, histo)
SORT_ALPHA = 'alpha'
SORT_COUNT = 'count'
SORT_OPTIONS = (SORT_ALPHA, SORT_COUNT)

def sort_alpha(histo: Histogram) -> CountAssocs:
    if False:
        for i in range(10):
            print('nop')
    ordered = sorted(histo.items(), key=lambda p: p[0])
    return list(ordered)

def sort_count(histo: Histogram) -> CountAssocs:
    if False:
        return 10
    ordered = sorted(histo.items(), key=lambda p: p[1], reverse=True)
    return list(ordered)
SORT_FUNCS = {SORT_ALPHA: sort_alpha, SORT_COUNT: sort_count}
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='\nRead and exec the contents of path and dump the distribution of\nbytecode for every function that was created.\n')
    parser.add_argument('path', help='path to file to exec')
    parser.add_argument('--sort', help='how to sort the output', default=SORT_COUNT, choices=SORT_OPTIONS)
    args = parser.parse_args()
    with open(args.path) as f:
        data = f.read()
    globals = {}
    exec(data, globals, globals)
    histo = {}
    for obj in globals.values():
        summarize(obj, histo)
    ordered = SORT_FUNCS[args.sort](histo)
    for (name, count) in ordered:
        print('%-20s %d' % (name, count))