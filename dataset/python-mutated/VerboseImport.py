"""
This module hooks into Python's import mechanism to print out all imports to
the standard output as they happen.
"""
__all__ = ()
import sys
oldimport = __import__
indentLevel = 0

def newimport(*args, **kw):
    if False:
        return 10
    global indentLevel
    fPrint = 0
    name = args[0]
    if name not in sys.modules:
        print(' ' * indentLevel + 'import ' + args[0])
        fPrint = 1
    indentLevel += 1
    result = oldimport(*args, **kw)
    indentLevel -= 1
    if fPrint:
        print(' ' * indentLevel + 'DONE: import ' + args[0])
    return result
__builtins__['__import__'] = newimport