"""A tool to inspect the binary size of a built binary file.

This script prints out a tree of symbols and their corresponding sizes, using
Linux's nm functionality.

Usage:

    python binary_size.py --             --target=/path/to/your/target/binary             [--nm_command=/path/to/your/custom/nm]             [--max_depth=10] [--min_size=1024]             [--color] 
To assist visualization, pass in '--color' to make the symbols color coded to
green, assuming that you have a xterm connection that supports color.
"""
import argparse
import subprocess
import sys

class Trie:
    """A simple class that represents a Trie."""

    def __init__(self, name):
        if False:
            while True:
                i = 10
        'Initializes a Trie object.'
        self.name = name
        self.size = 0
        self.dictionary = {}

def GetSymbolTrie(target, nm_command, max_depth):
    if False:
        print('Hello World!')
    'Gets a symbol trie with the passed in target.\n\n    Args:\n            target: the target binary to inspect.\n            nm_command: the command to run nm.\n            max_depth: the maximum depth to create the trie.\n    '
    proc = subprocess.Popen([nm_command, '--radix=d', '--size-sort', '--print-size', target], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (nm_out, _) = proc.communicate()
    if proc.returncode != 0:
        print('NM command failed. Output is as follows:')
        print(nm_out)
        sys.exit(1)
    proc = subprocess.Popen(['c++filt'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate(input=nm_out)
    if proc.returncode != 0:
        print('c++filt failed. Output is as follows:')
        print(out)
        sys.exit(1)
    data = []
    for line in out.split('\n'):
        if line:
            content = line.split(' ')
            if len(content) < 4:
                continue
            data.append([int(content[1]), ' '.join(content[3:])])
    symbol_trie = Trie('')
    for (size, name) in data:
        curr = symbol_trie
        for c in name:
            if c not in curr.dictionary:
                curr.dictionary[c] = Trie(curr.name + c)
            curr = curr.dictionary[c]
            curr.size += size
            if len(curr.name) > max_depth:
                break
    symbol_trie.size = sum((t.size for t in symbol_trie.dictionary.values()))
    return symbol_trie

def MaybeAddColor(s, color):
    if False:
        while True:
            i = 10
    'Wrap the input string to the xterm green color, if color is set.\n    '
    if color:
        return '\x1b[92m{0}\x1b[0m'.format(s)
    else:
        return s

def ReadableSize(num):
    if False:
        i = 10
        return i + 15
    'Get a human-readable size.'
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(num) <= 1024.0:
            return '%3.2f%s' % (num, unit)
        num /= 1024.0
    return '%.1f TB' % (num,)

def PrintTrie(trie, prefix, max_depth, min_size, color):
    if False:
        while True:
            i = 10
    'Prints the symbol trie in a readable manner.\n    '
    if len(trie.name) == max_depth or not trie.dictionary.keys():
        if trie.size > min_size:
            print('{0}{1} {2}'.format(prefix, MaybeAddColor(trie.name, color), ReadableSize(trie.size)))
    elif len(trie.dictionary.keys()) == 1:
        PrintTrie(trie.dictionary.values()[0], prefix, max_depth, min_size, color)
    elif trie.size > min_size:
        print('{0}{1} {2}'.format(prefix, MaybeAddColor(trie.name, color), ReadableSize(trie.size)))
        keys_with_sizes = [(k, trie.dictionary[k].size) for k in trie.dictionary.keys()]
        keys_with_sizes.sort(key=lambda x: x[1])
        for (k, _) in keys_with_sizes[::-1]:
            PrintTrie(trie.dictionary[k], prefix + ' |', max_depth, min_size, color)

def main(argv):
    if False:
        i = 10
        return i + 15
    if not sys.platform.startswith('linux'):
        raise RuntimeError('Currently this tool only supports Linux.')
    parser = argparse.ArgumentParser(description='Tool to inspect binary size.')
    parser.add_argument('--max_depth', type=int, default=10, help='The maximum depth to print the symbol tree.')
    parser.add_argument('--min_size', type=int, default=1024, help='The mininum symbol size to print.')
    parser.add_argument('--nm_command', type=str, default='nm', help='The path to the nm command that the tool needs.')
    parser.add_argument('--color', action='store_true', help='If set, use ascii color for output.')
    parser.add_argument('--target', type=str, help='The binary target to inspect.')
    args = parser.parse_args(argv)
    if not args.target:
        raise RuntimeError('You must specify a target to inspect.')
    symbol_trie = GetSymbolTrie(args.target, args.nm_command, args.max_depth)
    PrintTrie(symbol_trie, '', args.max_depth, args.min_size, args.color)
if __name__ == '__main__':
    main(sys.argv[1:])