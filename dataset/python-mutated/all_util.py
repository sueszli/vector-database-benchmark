"""Generate __all__ from a module docstring."""
import re as _re
import sys as _sys
from tensorflow.python.util import tf_inspect as _tf_inspect
_reference_pattern = _re.compile('^@@(\\w+)$', flags=_re.MULTILINE)

def make_all(module_name, doc_string_modules=None):
    if False:
        print('Hello World!')
    'Generates `__all__` from the docstring of one or more modules.\n\n  Usage: `make_all(__name__)` or\n  `make_all(__name__, [sys.modules(__name__), other_module])`. The doc string\n  modules must each a docstring, and `__all__` will contain all symbols with\n  `@@` references, where that symbol currently exists in the module named\n  `module_name`.\n\n  Args:\n    module_name: The name of the module (usually `__name__`).\n    doc_string_modules: a list of modules from which to take docstring.\n    If None, then a list containing only the module named `module_name` is used.\n\n  Returns:\n    A list suitable for use as `__all__`.\n  '
    if doc_string_modules is None:
        doc_string_modules = [_sys.modules[module_name]]
    cur_members = set((name for (name, _) in _tf_inspect.getmembers(_sys.modules[module_name])))
    results = set()
    for doc_module in doc_string_modules:
        results.update([m.group(1) for m in _reference_pattern.finditer(doc_module.__doc__) if m.group(1) in cur_members])
    return list(results)
_HIDDEN_ATTRIBUTES = {}

def reveal_undocumented(symbol_name, target_module=None):
    if False:
        for i in range(10):
            print('nop')
    'Reveals a symbol that was previously removed by `remove_undocumented`.\n\n  This should be used by tensorflow internal tests only. It explicitly\n  defeats the encapsulation afforded by `remove_undocumented`.\n\n  It throws an exception when the symbol was not hidden in the first place.\n\n  Args:\n    symbol_name: a string representing the full absolute path of the symbol.\n    target_module: if specified, the module in which to restore the symbol.\n  '
    if symbol_name not in _HIDDEN_ATTRIBUTES:
        raise LookupError('Symbol %s is not a hidden symbol' % symbol_name)
    symbol_basename = symbol_name.split('.')[-1]
    (original_module, attr_value) = _HIDDEN_ATTRIBUTES[symbol_name]
    if not target_module:
        target_module = original_module
    setattr(target_module, symbol_basename, attr_value)

def remove_undocumented(module_name, allowed_exception_list=None, doc_string_modules=None):
    if False:
        print('Hello World!')
    'Removes symbols in a module that are not referenced by a docstring.\n\n  Args:\n    module_name: the name of the module (usually `__name__`).\n    allowed_exception_list: a list of names that should not be removed.\n    doc_string_modules: a list of modules from which to take the docstrings.\n    If None, then a list containing only the module named `module_name` is used.\n\n    Furthermore, if a symbol previously added with `add_to_global_allowlist`,\n    then it will always be allowed. This is useful for internal tests.\n\n  Returns:\n    None\n  '
    current_symbols = set(dir(_sys.modules[module_name]))
    should_have = make_all(module_name, doc_string_modules)
    should_have += allowed_exception_list or []
    extra_symbols = current_symbols - set(should_have)
    target_module = _sys.modules[module_name]
    for extra_symbol in extra_symbols:
        if extra_symbol.startswith('_'):
            continue
        fully_qualified_name = module_name + '.' + extra_symbol
        _HIDDEN_ATTRIBUTES[fully_qualified_name] = (target_module, getattr(target_module, extra_symbol))
        delattr(target_module, extra_symbol)
__all__ = ['make_all', 'remove_undocumented', 'reveal_undocumented']