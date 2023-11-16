import warnings
from scipy._lib import doccer
__all__ = ['docformat', 'inherit_docstring_from', 'indentcount_lines', 'filldoc', 'unindent_dict', 'unindent_string', 'extend_notes_in_docstring', 'replace_notes_in_docstring']

def __dir__():
    if False:
        print('Hello World!')
    return __all__

def __getattr__(name):
    if False:
        while True:
            i = 10
    if name not in __all__:
        raise AttributeError(f'scipy.misc.doccer is deprecated and has no attribute {name}.')
    warnings.warn('The `scipy.misc.doccer` namespace is deprecated and will be removed in SciPy v2.0.0.', category=DeprecationWarning, stacklevel=2)
    return getattr(doccer, name)