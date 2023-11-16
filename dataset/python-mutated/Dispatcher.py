"""
Internal method dispatcher for Microsoft Visual C/C++.

MSVC modules can register their module (register_modulename) and individual
classes (register_class) with the method dispatcher during initialization. MSVC
modules tend to be registered immediately after the Dispatcher import near the
top of the file. Methods in the MSVC modules can be invoked indirectly without
having to hard-code the method calls effectively decoupling the upstream module
with the downstream modules:

The reset method dispatches calls to all registered objects with a reset method
and/or a _reset method. The reset methods are used to restore data structures
to their initial state for testing purposes. Typically, this involves clearing
cached values.

The verify method dispatches calls to all registered objects with a verify
method and/or a _verify method. The verify methods are used to check that
initialized data structures distributed across multiple modules are internally
consistent.  An exception is raised when a verification constraint violation
is detected.  Typically, this verifies that initialized dictionaries support
all of the requisite keys as new versions are added.
"""
import sys
from ..common import debug
_refs = []

def register_modulename(modname):
    if False:
        print('Hello World!')
    module = sys.modules[modname]
    _refs.append(module)

def register_class(ref):
    if False:
        while True:
            i = 10
    _refs.append(ref)

def reset():
    if False:
        print('Hello World!')
    debug('')
    for ref in _refs:
        for method in ['reset', '_reset']:
            if not hasattr(ref, method) or not callable(getattr(ref, method, None)):
                continue
            debug('call %s.%s()', ref.__name__, method)
            func = getattr(ref, method)
            func()

def verify():
    if False:
        return 10
    debug('')
    for ref in _refs:
        for method in ['verify', '_verify']:
            if not hasattr(ref, method) or not callable(getattr(ref, method, None)):
                continue
            debug('call %s.%s()', ref.__name__, method)
            func = getattr(ref, method)
            func()