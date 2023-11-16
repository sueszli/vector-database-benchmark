import os
import argparse
import sys
parser = argparse.ArgumentParser()
parser.add_argument('--tsv', required=True, type=str)
parser.add_argument('--no-skip', action='store_true')
parser.add_argument('--keep', action='store_true')
params = parser.parse_args()

def get_fname(line):
    if False:
        for i in range(10):
            print('nop')
    p = os.path.basename(line.split('\t')[0])
    p = os.path.splitext(p)[0]
    return p
seen = set()
with open(params.tsv) as f:
    if not params.no_skip:
        root = next(f).rstrip()
    for line in f:
        seen.add(get_fname(line))
for (i, line) in enumerate(sys.stdin):
    exists = get_fname(line) in seen
    keep = exists and params.keep or (not exists and (not params.keep))
    if i == 0 or keep:
        print(line, end='')