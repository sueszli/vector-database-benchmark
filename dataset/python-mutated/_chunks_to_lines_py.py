"""The python implementation of chunks_to_lines"""
from __future__ import absolute_import

def chunks_to_lines(chunks):
    if False:
        i = 10
        return i + 15
    'Re-split chunks into simple lines.\n\n    Each entry in the result should contain a single newline at the end. Except\n    for the last entry which may not have a final newline. If chunks is already\n    a simple list of lines, we return it directly.\n\n    :param chunks: An list/tuple of strings. If chunks is already a list of\n        lines, then we will return it as-is.\n    :return: A list of strings.\n    '
    last_no_newline = False
    for chunk in chunks:
        if last_no_newline:
            break
        if not chunk:
            break
        elif '\n' in chunk[:-1]:
            break
        elif chunk[-1] != '\n':
            last_no_newline = True
    else:
        return chunks
    from bzrlib import osutils
    return osutils._split_lines(''.join(chunks))