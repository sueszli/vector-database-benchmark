__license__ = 'GPL v3'
__copyright__ = '2008, Kovid Goyal <kovid at kovidgoyal.net>'
import re
from collections import defaultdict
PREAMBLE = '.. _templaterefcalibre-{}:\n\nReference for all built-in template language functions\n========================================================\n\nHere, we document all the built-in functions available in the calibre template\nlanguage. Every function is implemented as a class in python and you can click\nthe source links to see the source code, in case the documentation is\ninsufficient. The functions are arranged in logical groups by type.\n\n.. contents::\n    :depth: 2\n    :local:\n\n.. module:: calibre.utils.formatter_functions\n\n'
CATEGORY_TEMPLATE = '{category}\n{dashes}\n\n'
FUNCTION_TEMPLATE = '{fs}\n{hats}\n\n.. autoclass:: {cn}\n\n'
POSTAMBLE = "\nAPI of the Metadata objects\n----------------------------\n\nThe python implementation of the template functions is passed in a Metadata\nobject. Knowing it's API is useful if you want to define your own template\nfunctions.\n\n.. module:: calibre.ebooks.metadata.book.base\n\n.. autoclass:: Metadata\n   :members:\n   :member-order: bysource\n\n.. data:: STANDARD_METADATA_FIELDS\n\n    The set of standard metadata fields.\n\n.. literalinclude:: ../../../src/calibre/ebooks/metadata/book/__init__.py\n   :lines: 7-\n"

def generate_template_language_help(language):
    if False:
        i = 10
        return i + 15
    from calibre.utils.formatter_functions import formatter_functions
    pat = re.compile('\\)`{0,2}\\s*-{1,2}')
    funcs = defaultdict(dict)
    for func in formatter_functions().get_builtins().values():
        class_name = func.__class__.__name__
        func_sig = getattr(func, 'doc')
        m = pat.search(func_sig)
        if m is None:
            print('No signature for template function ', class_name)
            continue
        func_sig = func_sig[:m.start() + 1].strip('`')
        func_cat = getattr(func, 'category')
        funcs[func_cat][func_sig] = class_name
    output = PREAMBLE.format(language)
    cats = sorted(funcs.keys())
    for cat in cats:
        output += CATEGORY_TEMPLATE.format(category=cat, dashes='-' * len(cat))
        entries = [k for k in sorted(funcs[cat].keys())]
        for entry in entries:
            output += FUNCTION_TEMPLATE.format(fs=entry, cn=funcs[cat][entry], hats='^' * len(entry))
    output += POSTAMBLE
    return output
if __name__ == '__main__':
    generate_template_language_help()