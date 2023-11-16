"""Sort a simple YAML file, keeping blocks of comments and definitions
together.

We assume a strict subset of YAML that looks like:

    # block of header comments
    # here that should always
    # be at the top of the file

    # optional comments
    # can go here
    key: value
    key: value

    key: value

In other words, we don't sort deeper than the top layer, and might corrupt
complicated YAML files.
"""
from __future__ import annotations
import argparse
from typing import Sequence
QUOTES = ["'", '"']

def sort(lines: list[str]) -> list[str]:
    if False:
        return 10
    'Sort a YAML file in alphabetical order, keeping blocks together.\n\n    :param lines: array of strings (without newlines)\n    :return: sorted array of strings\n    '
    lines = list(lines)
    new_lines = parse_block(lines, header=True)
    for block in sorted(parse_blocks(lines), key=first_key):
        if new_lines:
            new_lines.append('')
        new_lines.extend(block)
    return new_lines

def parse_block(lines: list[str], header: bool=False) -> list[str]:
    if False:
        return 10
    'Parse and return a single block, popping off the start of `lines`.\n\n    If parsing a header block, we stop after we reach a line that is not a\n    comment. Otherwise, we stop after reaching an empty line.\n\n    :param lines: list of lines\n    :param header: whether we are parsing a header block\n    :return: list of lines that form the single block\n    '
    block_lines = []
    while lines and lines[0] and (not header or lines[0].startswith('#')):
        block_lines.append(lines.pop(0))
    return block_lines

def parse_blocks(lines: list[str]) -> list[list[str]]:
    if False:
        i = 10
        return i + 15
    'Parse and return all possible blocks, popping off the start of `lines`.\n\n    :param lines: list of lines\n    :return: list of blocks, where each block is a list of lines\n    '
    blocks = []
    while lines:
        if lines[0] == '':
            lines.pop(0)
        else:
            blocks.append(parse_block(lines))
    return blocks

def first_key(lines: list[str]) -> str:
    if False:
        while True:
            i = 10
    "Returns a string representing the sort key of a block.\n\n    The sort key is the first YAML key we encounter, ignoring comments, and\n    stripping leading quotes.\n\n    >>> print(test)\n    # some comment\n    'foo': true\n    >>> first_key(test)\n    'foo'\n    "
    for line in lines:
        if line.startswith('#'):
            continue
        if any((line.startswith(quote) for quote in QUOTES)):
            return line[1:]
        return line
    else:
        return ''

def main(argv: Sequence[str] | None=None) -> int:
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='*', help='Filenames to fix')
    args = parser.parse_args(argv)
    retval = 0
    for filename in args.filenames:
        with open(filename, 'r+') as f:
            lines = [line.rstrip() for line in f.readlines()]
            new_lines = sort(lines)
            if lines != new_lines:
                print(f'Fixing file `{filename}`')
                f.seek(0)
                f.write('\n'.join(new_lines) + '\n')
                f.truncate()
                retval = 1
    return retval
if __name__ == '__main__':
    raise SystemExit(main())