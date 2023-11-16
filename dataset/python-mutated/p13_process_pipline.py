"""
Topic: sample
Desc : 
"""
import os
import fnmatch
import gzip
import bz2
import re

def gen_find(filepat, top):
    if False:
        for i in range(10):
            print('nop')
    '\n    Find all filenames in a directory tree that match a shell wildcard pattern\n    '
    for (path, dirlist, filelist) in os.walk(top):
        for name in fnmatch.filter(filelist, filepat):
            yield os.path.join(path, name)

def gen_opener(filenames):
    if False:
        i = 10
        return i + 15
    '\n    Open a sequence of filenames one at a time producing a file object.\n    The file is closed immediately when proceeding to the next iteration.\n    '
    for filename in filenames:
        if filename.endswith('.gz'):
            f = gzip.open(filename, 'rt')
        elif filename.endswith('.bz2'):
            f = bz2.open(filename, 'rt')
        else:
            f = open(filename, 'rt')
        yield f
        f.close()

def gen_concatenate(iterators):
    if False:
        i = 10
        return i + 15
    '\n    Chain a sequence of iterators together into a single sequence.\n    '
    for it in iterators:
        yield from it

def gen_grep(pattern, lines):
    if False:
        while True:
            i = 10
    '\n    Look for a regex pattern in a sequence of lines\n    '
    pat = re.compile(pattern)
    for line in lines:
        if pat.search(line):
            yield line

def process_pipline():
    if False:
        for i in range(10):
            print('nop')
    lognames = gen_find('access-log*', 'www')
    files = gen_opener(lognames)
    lines = gen_concatenate(files)
    pylines = gen_grep('(?i)python', lines)
    for line in pylines:
        print(line)
    lognames = gen_find('access-log*', 'www')
    files = gen_opener(lognames)
    lines = gen_concatenate(files)
    pylines = gen_grep('(?i)python', lines)
    bytecolumn = (line.rsplit(None, 1)[1] for line in pylines)
    bytes = (int(x) for x in bytecolumn if x != '-')
    print('Total', sum(bytes))
if __name__ == '__main__':
    process_pipline()