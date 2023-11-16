from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
from renpy.compat import PY2, basestring, bchr, bord, chr, open, pystr, range, round, str, tobytes, unicode
import collections
import renpy
import os
missing = collections.defaultdict(list)

def report_missing(target, filename, position):
    if False:
        print('Hello World!')
    '\n    Reports that the call statement ending at `position` in `filename`\n    is missing a from clause.\n    '
    missing[filename].append((position, target))
new_labels = set()

def generate_label(target):
    if False:
        for i in range(10):
            print('nop')
    '\n    Generate a reasonable and unique new label for a call to `target`.\n    '
    target = target.replace('.', '_')
    n = 0
    while True:
        if n:
            label = '_call_{}_{}'.format(target, n)
        else:
            label = '_call_{}'.format(target)
        if not renpy.exports.has_label(label) and (not label in new_labels):
            break
        n += 1
    new_labels.add(label)
    return label

def process_file(fn):
    if False:
        i = 10
        return i + 15
    '\n    Adds missing from clauses to `fn`.\n    '
    if not os.path.exists(fn):
        return
    edits = missing[fn]
    edits.sort()
    with open(fn, 'rb') as f:
        data = f.read().decode('utf-8')
    consumed = 0
    output = u''
    for (position, target) in edits:
        output += data[consumed:position]
        consumed = position
        output += ' from {}'.format(generate_label(target))
    output += data[consumed:]
    with open(fn + '.new', 'wb') as f:
        f.write(output.encode('utf-8'))
    try:
        os.unlink(fn + '.bak')
    except Exception:
        pass
    os.rename(fn, fn + '.bak')
    os.rename(fn + '.new', fn)

def add_from():
    if False:
        i = 10
        return i + 15
    renpy.arguments.takes_no_arguments('Adds from clauses to call statements that are missing them.')
    for fn in missing:
        if fn.startswith(renpy.config.gamedir):
            process_file(fn)
    return False
renpy.arguments.register_command('add_from', add_from)