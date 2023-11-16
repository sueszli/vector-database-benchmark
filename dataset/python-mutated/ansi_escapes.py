"""
Provides ANSI escape sequences for coloring/formatting output in ANSI terminals.
"""
from __future__ import absolute_import
import os
import re
colors = {'black': u'\x1b[30m', 'red': u'\x1b[31m', 'green': u'\x1b[32m', 'yellow': u'\x1b[33m', 'blue': u'\x1b[34m', 'magenta': u'\x1b[35m', 'cyan': u'\x1b[36m', 'white': u'\x1b[37m', 'grey': u'\x1b[90m', 'bold': u'\x1b[1m'}
aliases = {'untested': 'cyan', 'undefined': 'yellow', 'pending': 'yellow', 'executing': 'grey', 'failed': 'red', 'passed': 'green', 'outline': 'cyan', 'skipped': 'cyan', 'comments': 'grey', 'tag': 'cyan'}
escapes = {'reset': u'\x1b[0m', 'up': u'\x1b[1A'}
_ANSI_ESCAPE_PATTERN = re.compile(u'\x1b\\[\\d+[mA]', re.UNICODE)

def _setup_module():
    if False:
        for i in range(10):
            print('nop')
    'Setup the remaining ANSI color aliases and ANSI escape sequences.\n\n    .. note:: May modify/extend the module attributes:\n\n        * :attr:`aliases`\n        * :attr:`escapes`\n    '
    if 'GHERKIN_COLORS' in os.environ:
        new_aliases = [p.split('=') for p in os.environ['GHERKIN_COLORS'].split(':')]
        aliases.update(dict(new_aliases))
    for alias in aliases:
        escapes[alias] = ''.join([colors[c] for c in aliases[alias].split(',')])
        arg_alias = alias + '_arg'
        arg_seq = aliases.get(arg_alias, aliases[alias] + ',bold')
        escapes[arg_alias] = ''.join([colors[c] for c in arg_seq.split(',')])
_setup_module()

def up(n):
    if False:
        print('Hello World!')
    return u'\x1b[%dA' % n

def strip_escapes(text):
    if False:
        while True:
            i = 10
    'Removes ANSI escape sequences from text (if any are contained).\n\n    :param text: Text that may or may not contain ANSI escape sequences.\n    :return: Text without ANSI escape sequences.\n    '
    return _ANSI_ESCAPE_PATTERN.sub('', text)

def use_ansi_escape_colorbold_composites():
    if False:
        print('Hello World!')
    'Patch for "sphinxcontrib-ansi" to process the following ANSI escapes\n    correctly (set-color set-bold sequences):\n\n        ESC[{color}mESC[1m  => ESC[{color};1m\n\n    Reapply aliases to ANSI escapes mapping.\n    '
    color_codes = {}
    for (color_name, color_escape) in colors.items():
        color_code = color_escape.replace(u'\x1b[', u'').replace(u'm', u'')
        color_codes[color_name] = color_code
    for alias in aliases:
        parts = [color_codes[c] for c in aliases[alias].split(',')]
        composite_escape = u'\x1b[{0}m'.format(u';'.join(parts))
        escapes[alias] = composite_escape
        arg_alias = alias + '_arg'
        arg_seq = aliases.get(arg_alias, aliases[alias] + ',bold')
        parts = [color_codes[c] for c in arg_seq.split(',')]
        composite_escape = u'\x1b[{0}m'.format(u';'.join(parts))
        escapes[arg_alias] = composite_escape