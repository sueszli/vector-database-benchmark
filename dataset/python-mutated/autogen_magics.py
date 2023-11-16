from pathlib import Path
from IPython.core.alias import Alias
from IPython.core.interactiveshell import InteractiveShell
from IPython.core.magic import MagicAlias
from IPython.utils.text import dedent, indent
shell = InteractiveShell.instance()
magics = shell.magics_manager.magics

def _strip_underline(line):
    if False:
        i = 10
        return i + 15
    chars = set(line.strip())
    if len(chars) == 1 and ('-' in chars or '=' in chars):
        return ''
    else:
        return line

def format_docstring(func):
    if False:
        return 10
    docstring = (func.__doc__ or 'Undocumented').rstrip()
    docstring = indent(dedent(docstring))
    lines = [_strip_underline(l) for l in docstring.splitlines()]
    return '\n'.join(lines)
output = ['Line magics', '===========', '']

def sortkey(s):
    if False:
        while True:
            i = 10
    return s[0].lower()
for (name, func) in sorted(magics['line'].items(), key=sortkey):
    if isinstance(func, Alias) or isinstance(func, MagicAlias):
        continue
    output.extend(['.. magic:: {}'.format(name), '', format_docstring(func), ''])
output.extend(['Cell magics', '===========', ''])
for (name, func) in sorted(magics['cell'].items(), key=sortkey):
    if name == '!':
        continue
    if func == magics['line'].get(name, 'QQQP'):
        continue
    if isinstance(func, MagicAlias):
        continue
    output.extend(['.. cellmagic:: {}'.format(name), '', format_docstring(func), ''])
src_path = Path(__file__).parent
dest = src_path.joinpath('source', 'interactive', 'magics-generated.txt')
dest.write_text('\n'.join(output), encoding='utf-8')