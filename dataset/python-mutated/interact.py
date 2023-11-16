"""This module enables interactive mode in Python Fire.

It uses IPython as an optional dependency. When IPython is installed, the
interactive flag will use IPython's REPL. When IPython is not installed, the
interactive flag will start a Python REPL with the builtin `code` module's
InteractiveConsole class.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect

def Embed(variables, verbose=False):
    if False:
        print('Hello World!')
    "Drops into a Python REPL with variables available as local variables.\n\n  Args:\n    variables: A dict of variables to make available. Keys are variable names.\n        Values are variable values.\n    verbose: Whether to include 'hidden' members, those keys starting with _.\n  "
    print(_AvailableString(variables, verbose))
    try:
        _EmbedIPython(variables)
    except ImportError:
        _EmbedCode(variables)

def _AvailableString(variables, verbose=False):
    if False:
        while True:
            i = 10
    "Returns a string describing what objects are available in the Python REPL.\n\n  Args:\n    variables: A dict of the object to be available in the REPL.\n    verbose: Whether to include 'hidden' members, those keys starting with _.\n  Returns:\n    A string fit for printing at the start of the REPL, indicating what objects\n    are available for the user to use.\n  "
    modules = []
    other = []
    for (name, value) in variables.items():
        if not verbose and name.startswith('_'):
            continue
        if '-' in name or '/' in name:
            continue
        if inspect.ismodule(value):
            modules.append(name)
        else:
            other.append(name)
    lists = [('Modules', modules), ('Objects', other)]
    liststrs = []
    for (name, varlist) in lists:
        if varlist:
            liststrs.append('{name}: {items}'.format(name=name, items=', '.join(sorted(varlist))))
    return 'Fire is starting a Python REPL with the following objects:\n{liststrs}\n'.format(liststrs='\n'.join(liststrs))

def _EmbedIPython(variables, argv=None):
    if False:
        i = 10
        return i + 15
    'Drops into an IPython REPL with variables available for use.\n\n  Args:\n    variables: A dict of variables to make available. Keys are variable names.\n        Values are variable values.\n    argv: The argv to use for starting ipython. Defaults to an empty list.\n  '
    import IPython
    argv = argv or []
    IPython.start_ipython(argv=argv, user_ns=variables)

def _EmbedCode(variables):
    if False:
        while True:
            i = 10
    import code
    code.InteractiveConsole(variables).interact()