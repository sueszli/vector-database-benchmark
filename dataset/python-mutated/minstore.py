from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
from renpy.compat import PY2, basestring, bchr, bord, chr, open, pystr, range, round, str, tobytes, unicode
xrange = range
unicode = str
python_list = _list = list
python_dict = _dict = dict
python_object = _object = object
python_set = _set = set
_type = type
from renpy.revertable import RevertableList as __renpy__list__
list = __renpy__list__
from renpy.revertable import RevertableDict as __renpy__dict__
dict = __renpy__dict__
from renpy.revertable import RevertableSet as __renpy__set__
set = __renpy__set__
Set = __renpy__set__
from renpy.revertable import RevertableObject as object
from renpy.revertable import revertable_range as range
from renpy.revertable import revertable_sorted as sorted
import renpy.ui as ui
from renpy.translation import translate_string as __
from renpy.python import store_eval as eval
from renpy.display.core import absolute
import renpy
globals()['renpy'] = renpy.exports
_print = print

def print(*args, **kwargs):
    if False:
        return 10
    "\n    :undocumented:\n\n    This is a variant of the print function that forces a checkpoint\n    at the start of the next statement, so that it can't be rolled past.\n    "
    renpy.game.context().force_checkpoint = True
    _print(*args, **kwargs)

def _(s):
    if False:
        for i in range(10):
            print('nop')
    '\n    :undocumented: Documented directly in the .rst.\n\n    Flags a string as translatable, and returns it immediately. The string\n    will be translated when displayed by the text displayable.\n    '
    return s

def _p(s):
    if False:
        for i in range(10):
            print('nop')
    '\n    :doc: underscore_p\n    :name: _p\n\n    Reformats a string and flags it as translatable. The string will be\n    translated when displayed by the text displayable. This is intended\n    to define multi-line for use in strings, of the form::\n\n        define gui.about = _p("""\n            These two lines will be combined together\n            to form a long line.\n\n            This line will be separate.\n            """)\n\n    The reformatting is done by breaking the text up into lines,\n    removing whitespace from the start and end of each line. Blank lines\n    are removed at the end. When there is a blank line, a blank line is\n    inserted to separate paragraphs. The {p} tag breaks a line, but\n    doesn\'t add a blank one.\n\n    This can be used in a string translation, using the construct::\n\n        old "These two lines will be combined together to form a long line.\\n\\nThis line will be separate."\n        new _p("""\n            These two lines will be combined together\n            to form a long line. Bork bork bork.\n\n            This line will be separate. Bork bork bork.\n            """)\n    '
    import re
    lines = [i.strip() for i in s.split('\n')]
    if lines and (not lines[0]):
        lines.pop(0)
    if lines and (not lines[-1]):
        lines.pop()
    rv = ''
    para = []
    for l in lines:
        if not l:
            rv += ' '.join(para) + '\n\n'
            para = []
        elif re.search('\\{p[^}]*\\}$', l):
            para.append(l)
            rv += ' '.join(para)
            para = []
        else:
            para.append(l)
    rv += ' '.join(para)
    return rv

def input(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    :undocumented:\n    '
    raise Exception("The Python input and raw_input functions do not work with Ren'Py. Please use the renpy.input function instead.")
raw_input = input
__all__ = ['PY2', 'Set', '_', '__', '__renpy__dict__', '__renpy__list__', '__renpy__set__', '_dict', '_list', '_object', '_p', '_print', '_set', '_type', 'absolute', 'basestring', 'bchr', 'bord', 'dict', 'eval', 'input', 'list', 'object', 'open', 'print', 'python_dict', 'python_list', 'python_object', 'python_set', 'range', 'raw_input', 'set', 'sorted', 'str', 'tobytes', 'ui', 'unicode']
if PY2:
    __all__ = [bytes(i) for i in __all__]