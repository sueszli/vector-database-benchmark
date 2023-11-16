"""Converts vectors from text to a binary format for quicker manipulation.

Usage:

  text2bin.py -o <out> -v <vocab> vec1.txt [vec2.txt ...]

Optiona:

  -o <filename>, --output <filename>
    The name of the file into which the binary vectors are written.

  -v <filename>, --vocab <filename>
    The name of the file into which the vocabulary is written.

Description

This program merges one or more whitespace separated vector files into a single
binary vector file that can be used by downstream evaluation tools in this
directory ("wordsim.py" and "analogy").

If more than one vector file is specified, then the files must be aligned
row-wise (i.e., each line must correspond to the same embedding), and they must
have the same number of columns (i.e., be the same dimension).

"""
from itertools import izip
from getopt import GetoptError, getopt
import os
import struct
import sys
try:
    (opts, args) = getopt(sys.argv[1:], 'o:v:', ['output=', 'vocab='])
except GetoptError as e:
    (print >> sys.stderr, e)
    sys.exit(2)
opt_output = 'vecs.bin'
opt_vocab = 'vocab.txt'
for (o, a) in opts:
    if o in ('-o', '--output'):
        opt_output = a
    if o in ('-v', '--vocab'):
        opt_vocab = a

def go(fhs):
    if False:
        while True:
            i = 10
    fmt = None
    with open(opt_vocab, 'w') as vocab_out:
        with open(opt_output, 'w') as vecs_out:
            for lines in izip(*fhs):
                parts = [line.split() for line in lines]
                token = parts[0][0]
                if any((part[0] != token for part in parts[1:])):
                    raise IOError('vector files must be aligned')
                (print >> vocab_out, token)
                vec = [sum((float(x) for x in xs)) for xs in zip(*parts)[1:]]
                if not fmt:
                    fmt = struct.Struct('%df' % len(vec))
                vecs_out.write(fmt.pack(*vec))
if args:
    fhs = [open(filename) for filename in args]
    go(fhs)
    for fh in fhs:
        fh.close()
else:
    go([sys.stdin])